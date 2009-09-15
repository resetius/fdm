#!/usr/bin/perl
use DBI

print "drivers:\n";
@drivers = DBI->available_drivers;
for $i (@drivers) {
	print "$i\n";
}
print "---\n";

my $db_str   = "dbi:Pg:dbname=science";
my $username = "postgres";
my $password = "";

my $dbh = DBI->connect($db_str, $username, $password, {PrintError => 0});
if ($DBI::err != 0) {
	print $DBI::errstr . "\n";
	exit($DBI::err);
}

my $id = "CREATE SEQUENCE experiment_id_seq";
my $table1 = "CREATE TABLE experiment (
	id INT UNIQUE,
	name VARCHAR(256) -- link to experiment_table
)";
my $alter = "ALTER TABLE experiment ALTER COLUMN id SET DEFAULT NEXTVAL('experiment_id_seq')";

my $table2 = "CREATE TABLE barvortex_fdm (
	experiment_id INT, -- link to experiment
	domain_info VARCHAR(256),	
	mesh_w INT,
	mesh_h INT,
	tau FLOAT8,
	sigma FLOAT8,
	mu FLOAT8,
	k1 FLOAT8,
	k2 FLOAT8,
	theta FLOAT8,
	rp_info TEXT,
	coriolis_info TEXT,
	initial_info TEXT,
	build_info TEXT,
	other_info TEXT,
	cmd_info TEXT, -- command line
	calc_table VARCHAR(256)
)";

$dbh->do($id);
$dbh->do($table1);
$dbh->do($alter);
$dbh->do($table2);

my $uniq_name = $ARGV[1]; #md5 of input data
my $table3 = "CREATE TABLE barvortex_fdm_$uniq_name (
	t FLOAT8,
	v TEXT
)";
$dbh->do($table3);






