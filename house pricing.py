from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when
import matplotlib.pyplot as plt

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("HousingAnalysisBigData") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load data (assuming it's in a parquet file)
df = spark.read.parquet("housing_big_data.parquet")

# 1. Analysis of missing values
def analyze_missing_values(df):
    for column in df.columns:
        null_counts = df.select(when(col(column).isNull(), 1).otherwise(0).alias("is_null")).groupBy("is_null").count().collect()
        labels, values = zip(*[(row['is_null'], row['count']) for row in null_counts])
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title(f'Count of null values in {column}')
        plt.xlabel('Is null')
        plt.ylabel('Count')
        plt.show()

analyze_missing_values(df)

# 2. Compare price differences
def explore_features(df, features, target='SalePrice'):
    for feature in features:
        result = df.withColumn("year_difference", col("YrSold") - col(feature))
        
        # Convert to Pandas for visualization
        pandas_df = result.select("year_difference", target).toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(pandas_df["year_difference"], pandas_df[target])
        plt.xlabel(f'Year difference ({feature})')
        plt.ylabel(target)
        plt.title(f'Relationship between year difference in {feature} and {target}')
        plt.show()

time_features = ['YearBuilt', 'YearRemodAdd', 'YrSold']
explore_features(df, time_features)

# Prepare data for the model
feature_columns = [col for col in df.columns if col not in ['Id', 'SalePrice']]

# Create a feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df)

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
df_scaled = scaler.fit(df_assembled).transform(df_assembled)

# Split into training and test sets
train_data, test_data = df_scaled.randomSplit([0.8, 0.2], seed=42)

# Train linear regression model
lr = LinearRegression(featuresCol="scaledFeatures", labelCol="SalePrice", maxIter=10, regParam=0.3, elasticNetParam=1.0)
model = lr.fit(train_data)

# 3. Print the model formula
def print_model_formula(model, features):
    coefficients = model.coefficients
    intercept = model.intercept
    
    formula = f"SalePrice = {intercept:.2f}"
    for feat, coef in zip(features, coefficients):
        if coef != 0:
            formula += f" + ({coef:.2f} * {feat})"
    
    print("Model formula:")
    print(formula)

print_model_formula(model, feature_columns)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="r2").evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Visualize predictions vs actual values
pandas_predictions = predictions.select("SalePrice", "prediction").toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(pandas_predictions["SalePrice"], pandas_predictions["prediction"], alpha=0.5)
plt.plot([pandas_predictions["SalePrice"].min(), pandas_predictions["SalePrice"].max()], 
         [pandas_predictions["SalePrice"].min(), pandas_predictions["SalePrice"].max()], 
         'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predictions vs Actual Values")
plt.show()

# Close the Spark session
spark.stop()