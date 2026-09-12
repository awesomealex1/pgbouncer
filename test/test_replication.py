import signal
import socket
import struct
import subprocess
import time

import psycopg
import psycopg.errors
import pytest
from psycopg import sql

from .utils import PG_MAJOR_VERSION, WINDOWS, run


def _replication_startup_error(bouncer, **parameters):
    # libpq does not expose the SQLSTATE of connection failures through psycopg.
    # Read the wire response to check both the message and the SQLSTATE.
    parameters.setdefault("user", bouncer.default_user)
    parameters.setdefault("database", "user_passthrough")
    parameters.setdefault("replication", "database")
    encoded_parameters = b"".join(
        key.encode() + b"\0" + value.encode() + b"\0"
        for key, value in parameters.items()
    )
    payload = struct.pack("!I", 0x30000) + encoded_parameters + b"\0"

    timeout = bouncer.set_default_connection_options({})["connect_timeout"]
    with socket.create_connection(
        (bouncer.host, bouncer.port), timeout=timeout
    ) as sock:
        sock.sendall(struct.pack("!I", len(payload) + 4) + payload)
        with sock.makefile("rb") as stream:
            while message_type := stream.read(1):
                message_length = struct.unpack("!I", stream.read(4))[0]
                message = stream.read(message_length - 4)
                assert len(message) == message_length - 4
                if message_type == b"E":
                    error = {
                        field[:1].decode(): field[1:].decode()
                        for field in message.split(b"\0")
                        if field
                    }
                    assert error["S"] == "FATAL"
                    # Exactly one error, followed by connection closure.
                    assert stream.read(1) == b""
                    return error

    pytest.fail("server closed the connection without an ErrorResponse")


def test_logical_rep(bouncer):
    connect_args = {
        "dbname": "user_passthrough",
        "replication": "database",
        "user": "postgres",
        "application_name": "abc",
        "options": "-c enable_seqscan=off",
    }
    # Starting in PG10 you can do other commands over logical rep connections
    if PG_MAJOR_VERSION >= 10:
        bouncer.test(**connect_args)
        assert bouncer.sql_value("SHOW application_name", **connect_args) == "abc"
        assert bouncer.sql_value("SHOW enable_seqscan", **connect_args) == "off"
    bouncer.sql("IDENTIFY_SYSTEM", **connect_args)
    # Do a normal connection to the same pool, to ensure that that doesn't
    # break anything
    bouncer.test(dbname="user_passthrough", user="postgres")
    bouncer.sql("IDENTIFY_SYSTEM", **connect_args)


def test_logical_rep_auth_query(bouncer):
    connect_args = {
        "dbname": "pauthz",
        "replication": "database",
        "user": "pswcheck_not_in_auth_file",
        "application_name": "abc",
        "options": "-c enable_seqscan=off",
    }
    # Starting in PG10 you can do other commands over logical rep connections
    if PG_MAJOR_VERSION >= 10:
        bouncer.test(**connect_args)
        assert bouncer.sql_value("SHOW application_name", **connect_args) == "abc"
        assert bouncer.sql_value("SHOW enable_seqscan", **connect_args) == "off"
    bouncer.sql("IDENTIFY_SYSTEM", **connect_args)
    # Do a normal connection to the same pool, to ensure that that doesn't
    # break anything
    bouncer.test(dbname="user_passthrough", user="postgres")
    bouncer.sql("IDENTIFY_SYSTEM", **connect_args)


def test_logical_rep_unprivileged(bouncer):
    bouncer.admin("set server_login_retry = 60")
    if PG_MAJOR_VERSION < 10:
        expected_log = "no pg_hba.conf entry for replication connection"
    elif PG_MAJOR_VERSION < 16:
        expected_log = "must be superuser or replication role to start walsender"
    else:
        expected_log = "permission denied to start WAL sender"

    with bouncer.log_contains(rf"closing because: .*{expected_log}.*\(age", times=2):
        error = _replication_startup_error(bouncer, database="p0")

    assert expected_log in error["M"]
    assert error["C"] == ("28000" if PG_MAJOR_VERSION < 10 else "42501")

    # This role can still open ordinary connections in the same pool.
    bouncer.test()


def test_logical_rep_non_existing_database(bouncer):
    error = _replication_startup_error(bouncer, database="non_existing_pg_db")
    assert error["C"] == "3D000"
    assert error["M"] == 'database "non_existing_pg_db" does not exist'


@pytest.mark.parametrize("replication", ["database", "yes"])
def test_replication_startup_long_error(bouncer, replication):
    # Exercise the old 128-byte reason and 512-byte ErrorResponse limits, as
    # well as startup errors that exceed the default 4096-byte pkt_buf.
    value = "x" * 5000
    error = _replication_startup_error(
        bouncer, replication=replication, options=f"-c work_mem={value}"
    )
    assert error["C"] == "22023"
    assert error["M"] == f'invalid value for parameter "work_mem": "{value}"'


@pytest.mark.parametrize("replication", ["database", "yes"])
@pytest.mark.parametrize("cached_welcome", [False, True])
def test_replication_startup_errors_are_independent(
    bouncer, replication, cached_welcome
):
    bouncer.admin("set server_login_retry = 60")
    if cached_welcome:
        bouncer.test(dbname="user_passthrough")
        # Close the usable server without clearing the cached welcome message.
        bouncer.admin("pause user_passthrough")
        bouncer.admin("resume user_passthrough")

    # A cached welcome makes this a query error rather than a connection error.
    error_type = (
        psycopg.errors.InvalidParameterValue
        if cached_welcome
        else psycopg.OperationalError
    )
    # A bad startup option must not delay another replication attempt, or
    # leave an ordinary client with the previous replication client's error.
    for _ in range(2):
        with pytest.raises(error_type, match='invalid value for parameter "work_mem"'):
            bouncer.test(
                dbname="user_passthrough",
                replication=replication,
                options="-c work_mem=invalid",
            )

    bouncer.test(dbname="user_passthrough")
    bouncer.sql("IDENTIFY_SYSTEM", dbname="user_passthrough", replication=replication)


def test_replication_error_does_not_restart_pool_backoff(pg, bouncer):
    bouncer.admin("set verbose = 1")
    bouncer.admin("set server_login_retry = 2")
    bouncer.admin("set server_tls_sslmode = disable")
    pg.nossl_access("p0", "reject")
    pg.reload()
    with pytest.raises(
        psycopg.OperationalError, match="pg_hba.conf rejects connection"
    ):
        bouncer.test(dbname="user_passthrough")

    pg.reset_hba()
    pg.reload()
    time.sleep(2)

    # The original pool-wide backoff has expired. A replication-specific
    # failure must not restart its timer, even though the failure flag is set.
    with pytest.raises(psycopg.OperationalError, match="invalid value for parameter"):
        bouncer.test(
            dbname="user_passthrough",
            replication="database",
            options="-c work_mem=invalid",
        )

    with bouncer.log_contains("last failed, not launching new connection yet", times=0):
        bouncer.sql(
            "IDENTIFY_SYSTEM", dbname="user_passthrough", replication="database"
        )


@pytest.mark.parametrize("replication", ["database", "yes"])
def test_replication_startup_timeout_keeps_backoff(pg, bouncer, replication):
    pg.configure("pre_auth_delay to '5s'")
    pg.reload()
    # Send the StartupMessage before the timeout, so that the backend is
    # already marked as a replication connection when it is disconnected.
    bouncer.admin("set server_tls_sslmode = disable")
    bouncer.admin("set server_connect_timeout = 1")
    bouncer.admin("set server_login_retry = 60")

    with bouncer.log_contains("new connection to server", times=1):
        with pytest.raises(psycopg.OperationalError, match="connect timeout"):
            bouncer.test(dbname="user_passthrough", replication=replication)

        # Transport failures still apply to the whole pool: do not launch a
        # new backend for an ordinary client while the retry timer is active.
        with pytest.raises(psycopg.OperationalError, match="timeout"):
            bouncer.test(dbname="user_passthrough", connect_timeout=2)


def test_replication_auth_query_error_hidden(bouncer):
    # A replication client's auth_query still uses an ordinary server
    # connection, before client authentication has completed.
    bouncer.admin("set auth_query = 'SELECT * FROM no_such_auth_table'")
    with bouncer.log_contains('relation "no_such_auth_table" does not exist'):
        error = _replication_startup_error(
            bouncer,
            database="pauthz",
            user="pswcheck_not_in_auth_file",
        )

    assert error == {"S": "FATAL", "C": "08P01", "M": "bouncer config error"}


@pytest.mark.skipif(
    "PG_MAJOR_VERSION < 10", reason="logical replication was introduced in PG10"
)
def test_logical_rep_subscriber(bouncer):
    bouncer.admin("set pool_mode=transaction")

    # First write create a table and insert a row in the source database.
    # Also create the replication slot and publication
    bouncer.default_db = "user_passthrough"
    bouncer.create_schema("test_logical_rep_subscriber")
    bouncer.sql("CREATE TABLE test_logical_rep_subscriber.table(a int)")
    bouncer.sql("INSERT INTO test_logical_rep_subscriber.table values (1)")
    assert (
        bouncer.sql_value("SELECT count(*) FROM test_logical_rep_subscriber.table") == 1
    )

    bouncer.create_publication(
        "mypub", sql.SQL("FOR TABLE test_logical_rep_subscriber.table")
    )

    bouncer.create_logical_replication_slot("test_logical_rep_subscriber", "pgoutput")

    # Create an equivalent, but empty schema in the target database.
    # And setup the subscription
    bouncer.default_db = "user_passthrough2"
    bouncer.create_schema("test_logical_rep_subscriber")
    bouncer.sql("CREATE TABLE test_logical_rep_subscriber.table(a int)")
    conninfo = bouncer.make_conninfo(dbname="user_passthrough")
    bouncer.create_subscription(
        "mysub",
        sql.SQL("""
            CONNECTION {}
            PUBLICATION mypub
            WITH (slot_name=test_logical_rep_subscriber, create_slot=false)
        """).format(sql.Literal(conninfo)),
    )

    # The initial copy should now copy over the row
    time.sleep(2)
    assert (
        bouncer.sql_value("SELECT count(*) FROM test_logical_rep_subscriber.table") >= 1
    )

    # Insert another row and logical replication should replicate it correctly
    bouncer.sql(
        "INSERT INTO test_logical_rep_subscriber.table values (2)",
        dbname="user_passthrough",
    )
    time.sleep(2)
    assert (
        bouncer.sql_value("SELECT count(*) FROM test_logical_rep_subscriber.table") >= 2
    )


@pytest.mark.skipif(
    "WINDOWS", reason="MINGW does not have contrib package containing test_decoding"
)
def test_logical_rep_pg_recvlogical(bouncer):
    bouncer.default_db = "user_passthrough"
    bouncer.create_schema("test_logical_rep_pg_recvlogical")
    bouncer.sql("CREATE TABLE test_logical_rep_pg_recvlogical.table(a int)")
    bouncer.create_logical_replication_slot(
        "test_logical_rep_pg_recvlogical", "test_decoding"
    )
    process = subprocess.Popen(
        [
            "pg_recvlogical",
            "--dbname",
            bouncer.default_db,
            "--host",
            bouncer.host,
            "--port",
            str(bouncer.port),
            "--user",
            bouncer.default_user,
            "--slot=test_logical_rep_pg_recvlogical",
            "--file=-",
            "--no-loop",
            "--start",
        ],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    bouncer.sql("INSERT INTO test_logical_rep_pg_recvlogical.table values (1)")
    try:
        assert process.stdout.readline().startswith(b"BEGIN ")
        assert (
            process.stdout.readline()
            == b'table test_logical_rep_pg_recvlogical."table": INSERT: a[integer]:1\n'
        )
        assert process.stdout.readline().startswith(b"COMMIT ")
    finally:
        process.kill()
        process.communicate(timeout=5)


def test_physical_rep(bouncer):
    connect_args = {
        "dbname": "user_passthrough",
        "replication": "yes",
        "user": "postgres",
        "application_name": "abc",
        "options": "-c enable_seqscan=off",
    }
    # Starting in PG10 you can do SHOW commands
    if PG_MAJOR_VERSION >= 10:
        with pytest.raises(
            psycopg.errors.FeatureNotSupported,
            match="cannot execute SQL commands in WAL sender for physical replication",
        ):
            bouncer.test(**connect_args)
        assert bouncer.sql_value("SHOW application_name", **connect_args) == "abc"
        assert bouncer.sql_value("SHOW enable_seqscan", **connect_args) == "off"
    bouncer.sql("IDENTIFY_SYSTEM", **connect_args)
    # Do a normal connection to the same pool, to ensure that that doesn't
    # break anything
    bouncer.test(dbname="user_passthrough", user="postgres")
    bouncer.sql("IDENTIFY_SYSTEM", **connect_args)


def test_physcal_rep_unprivileged(bouncer):
    bouncer.admin("set server_login_retry = 60")
    expected_error = "no pg_hba.conf entry for replication connection from host"
    with bouncer.log_contains(rf"closing because: .*{expected_error}.*\(age", times=2):
        error = _replication_startup_error(bouncer, database="p0", replication="yes")

    assert expected_error in error["M"]
    assert error["C"] == "28000"

    # The replication-specific HBA rule must not block ordinary connections.
    bouncer.test()


@pytest.mark.skipif("PG_MAJOR_VERSION < 10", reason="pg_receivewal was added in PG10")
def test_physical_rep_pg_receivewal(bouncer, tmp_path):
    bouncer.default_db = "user_passthrough"
    bouncer.create_physical_replication_slot("test_physical_rep_pg_receivewal")
    wal_dump_dir = tmp_path / "wal-dump"
    wal_dump_dir.mkdir()

    process = subprocess.Popen(
        [
            "pg_receivewal",
            "--dbname",
            bouncer.make_conninfo(),
            "--slot=test_physical_rep_pg_receivewal",
            "--directory",
            str(wal_dump_dir),
        ],
    )
    time.sleep(3)

    if WINDOWS:
        process.terminate()
    else:
        process.send_signal(signal.SIGINT)
    process.communicate(timeout=5)

    if WINDOWS:
        assert process.returncode == 1
    else:
        assert process.returncode == 0

    children = list(wal_dump_dir.iterdir())
    assert len(children) > 0


def test_physical_rep_pg_basebackup(bouncer, tmp_path):
    bouncer.default_db = "user_passthrough"
    dump_dir = tmp_path / "db-dump"
    dump_dir.mkdir()

    run(
        [
            "pg_basebackup",
            "--dbname",
            bouncer.make_conninfo(),
            "--checkpoint=fast",
            "--pgdata",
            dump_dir,
        ],
    )
    children = list(dump_dir.iterdir())
    assert len(children) > 0
    print(children)


@pytest.mark.skipif(
    "PG_MAJOR_VERSION < 10",
    reason="normal SQL commands are only supported in PG10+ on logical replication connections",
)
async def test_replication_pool_size(pg, bouncer):
    connect_args = {
        "dbname": "user_passthrough_pool_size2",
        "replication": "database",
        "user": "postgres",
        "connect_timeout": 10,
    }
    start = time.time()
    await bouncer.asleep(0.5, times=10, **connect_args)
    assert time.time() - start > 2.5
    # Replication connections always get closed right away
    assert pg.connection_count("p0") == 0

    connect_args["dbname"] = "user_passthrough_pool_size5"
    start = time.time()
    await bouncer.asleep(0.5, times=10, **connect_args)
    assert time.time() - start > 1
    # Replication connections always get closed right away
    assert pg.connection_count("p0") == 0


@pytest.mark.skipif(
    "PG_MAJOR_VERSION < 10",
    reason="normal SQL commands are only supported in PG10+ on logical replication connections",
)
async def test_replication_pool_size_mixed_clients(bouncer):
    connect_args = {
        "dbname": "user_passthrough_pool_size2",
        "user": "postgres",
    }

    # Fill the pool with normal connections
    await bouncer.asleep(0.5, times=2, **connect_args)

    # Then try to open a replication connection and ensure that it causes
    # eviction of one of the normal connections
    with bouncer.log_contains("closing because: evicted"):
        bouncer.test(**connect_args, replication="database")
