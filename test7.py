import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, lead, round as spark_round, to_timestamp
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telematics_processor")

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Ingesting IoT driver telemetry batches...")

telemetry_records = [
    ("VIN_1001", "2026-03-01 10:00:00", 45.2, 77.21, 28.53),
    ("VIN_1001", "2026-03-01 10:01:00", 52.4, 77.22, 28.54),
    ("VIN_1001", "2026-03-01 10:02:00", 61.0, 77.23, 28.55),
    ("VIN_2002", "2026-03-01 10:00:30", 12.0, 77.10, 28.40),
    ("VIN_2002", "2026-03-01 10:01:30", 15.5, 77.11, 28.41)
]

columns = ["vehicle_id", "event_timestamp", "speed_kph", "longitude", "latitude"]
telemetry_df = spark.createDataFrame(telemetry_records, columns)

logger.info("Casting event_timestamp string column to TimestampType for deterministic ordering...")

# Cast the raw string timestamp to a proper TimestampType so that window ordering is
# chronologically correct instead of relying on lexical string ordering.
telemetry_df = telemetry_df.withColumn(
    "event_timestamp",
    to_timestamp(col("event_timestamp"), "yyyy-MM-dd HH:mm:ss")
)

logger.info("Computing telemetry delta metrics between successive pings...")

# FIXED: Window definition now includes orderBy('event_timestamp'), which is required
# by row_number() and lead() to produce deterministic, chronologically ordered results.
vehicle_window = Window.partitionBy("vehicle_id").orderBy("event_timestamp")

try:
    enriched_df = telemetry_df.withColumn("ping_seq", row_number().over(vehicle_window)) \
                              .withColumn("next_speed", lead("speed_kph", 1).over(vehicle_window))

    logger.info("Calculating instantaneous acceleration indices...")
    metrics_df = enriched_df.withColumn(
        "speed_delta", spark_round(col("next_speed") - col("speed_kph"), 2)
    )

    metrics_df.show()
    job.commit()
except Exception as e:
    logger.error("Failed while computing telemetry window metrics. Dumping schemas for diagnostics.")
    try:
        logger.error("telemetry_df schema:")
        telemetry_df.printSchema()
    except Exception:
        logger.error("Unable to print telemetry_df schema.")
    raise e
