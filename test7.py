import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.window import Window
from pyspark.sql.functions import col, coalesce, row_number, lead, round as spark_round
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

logger.info("Computing telemetry delta metrics between successive pings...")

# Fixed: added .orderBy("event_timestamp") so that row_number() and lead() produce
# deterministic, chronologically-ordered results within each vehicle partition.
vehicle_window = Window.partitionBy("vehicle_id").orderBy("event_timestamp")

enriched_df = telemetry_df.withColumn("ping_seq", row_number().over(vehicle_window)) \
                          .withColumn("next_speed", lead("speed_kph", 1).over(vehicle_window))

logger.info("Calculating instantaneous acceleration indices...")
# Fixed: wrapped next_speed with coalesce so that the last ping per vehicle
# (where lead returns null) produces a speed_delta of 0.0 instead of null.
metrics_df = enriched_df.withColumn(
    "speed_delta",
    spark_round(coalesce(col("next_speed"), col("speed_kph")) - col("speed_kph"), 2)
)

metrics_df.show()
job.commit()
