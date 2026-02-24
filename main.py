# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load datasets
logs = spark.read.csv("listening_logs.csv", header=True, inferSchema=True)
metadata = spark.read.csv("songs_metadata.csv", header=True, inferSchema=True)

joined = logs.join(metadata, on="song_id", how="inner")

# Task 1: User Favorite Genres
genre_counts = joined.groupBy("user_id", "genre").count()
window = Window.partitionBy("user_id").orderBy(col("count").desc())
favorite_genres = genre_counts \
    .withColumn("rank", rank().over(window)) \
    .filter(col("rank") == 1) \
    .drop("rank") \
    .orderBy("user_id")
favorite_genres.show(10)
favorite_genres.coalesce(1).write.mode("overwrite").csv("outputs/task1_favorite_genres", header=True)
print("Task 1 Done")

# Task 2: Average Listen Time
avg_listen = logs.groupBy("user_id") \
    .agg(round(avg("duration_sec"), 2).alias("avg_duration_sec")) \
    .orderBy("user_id")
avg_listen.show(10)
avg_listen.coalesce(1).write.mode("overwrite").csv("outputs/task2_avg_listen_time", header=True)
print("Task 2 Done")

# Task 3: Genre Loyalty Scores - Top 10
total_listens = logs.groupBy("user_id").agg(count("*").alias("total_listens"))
genre_listens = joined.groupBy("user_id", "genre").agg(count("*").alias("genre_count"))
top_genre = genre_listens \
    .withColumn("rank", rank().over(Window.partitionBy("user_id").orderBy(col("genre_count").desc()))) \
    .filter(col("rank") == 1).drop("rank")
loyalty = top_genre.join(total_listens, on="user_id") \
    .withColumn("loyalty_score", round((col("genre_count") / col("total_listens")) * 100, 2)) \
    .orderBy(col("loyalty_score").desc()) \
    .select("user_id", "genre", "genre_count", "total_listens", "loyalty_score")
loyalty.limit(10).show()
loyalty.limit(10).coalesce(1).write.mode("overwrite").csv("outputs/task3_genre_loyalty_top10", header=True)
print("Task 3 Done")

# Task 4: Users who listen between 12 AM and 5 AM
night_owls = logs.withColumn("hour", hour(col("timestamp"))) \
    .filter((col("hour") >= 0) & (col("hour") < 5)) \
    .select("user_id", "song_id", "timestamp", "hour") \
    .orderBy("user_id", "timestamp")
night_owls.show(10)
night_owls.coalesce(1).write.mode("overwrite").csv("outputs/task4_night_owl_users", header=True)
print("Task 4 Done")

spark.stop()
print("All tasks completed! Check outputs/ folder.")