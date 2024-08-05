# Import delle librerie necessarie
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as sql_sum, udf, lower, round as sql_round
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from graphframes import GraphFrame
import os, sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Creazione di una sessione Spark
spark = SparkSession.builder \
    .appName("Retweet Network Analysis") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# Definizione della classe per l'analisi della rete di retweet
class RetweetNetworkAnalysis:
    """
    Questa classe gestisce il caricamento dei dati dei tweet,
    la costruzione del grafo della rete di retweet e il calcolo delle metriche di centralità.
    """

    def __init__(self, spark):
        """
        Inizializza l'oggetto RetweetNetworkAnalysis con una sessione Spark.

        Args:
            spark (SparkSession): La sessione Spark da utilizzare per le operazioni.
        """
        self.spark = spark

    def load_data(self, file_path):
        """
        Carica i dati dei tweet da un file JSON, seleziona e rinomina le colonne necessarie,
        estrae gli utenti distinti e crea gli edge per i retweet.

        Args:
            file_path (str): Il percorso del file JSON contenente i dati dei tweet.

        Returns:
            DataFrame: DataFrame degli utenti distinti.
            DataFrame: DataFrame degli edge della rete di retweet.
        """
        logger.info("Caricamento dei dati...")
        tweets_df = self.spark.read.json(file_path)

        # Rimozione duplicati
        tweets_df = tweets_df.dropDuplicates()

        tweets_df = tweets_df.filter(col("text").isNotNull())

        rows_tweets_df_no_dupl = tweets_df.filter(col("text").isNotNull()).count()  # 999,673
        print(f"Numero righe tweets_df {rows_tweets_df_no_dupl}")

        # Selezione e rinomina le colonne necessarie
        tweets_df = tweets_df.select("id", col("user.name").alias("us"), col("entities.hashtags.text").alias("hashtags"), "text", col("retweeted_status.user.name").alias("rus"))

        # Estrazione utenti distinti e cache per migliorare le prestazioni
        users_df = tweets_df.select(col("us").alias("id")).union(tweets_df.select(col("rus"))).distinct().cache()
        # users_df.show(5)
        logger.info(f"Numero di utenti distinti: {users_df.count()}") # 493,904

        # Creazione degli edge tra utenti sulla base dei retweet
        edges_df = tweets_df.select(col("us").alias("src"), col("rus").alias("dst")).dropna().cache()
        logger.info(f"Numero di edge: {edges_df.count()}") # 444,612

        return users_df, edges_df

    def calculate_closeness_centrality(self, graph, vertex):
        """
        Calcola la centralità di closeness per un dato vertice nel grafo.

        Args:
            graph (GraphFrame): Il GraphFrame contenente la rete di retweet.
            vertex (str): L'id del vertice per cui calcolare la centralità di closeness.

        Returns:
            tuple: Una tupla contenente l'id del vertice e la sua centralità di closeness.
        """
        print(f"vertex {vertex}")

        # Calcolo dei percorsi più brevi
        shortest_paths = graph.shortestPaths(landmarks=["vertex"]).select("id", "distances")

        # Estrazione delle distanze e conversione in DataFrame
        distances = shortest_paths.rdd.map(lambda row: (row["id"], float(row["distances"].get(vertex, float('inf')))))

        schema = StructType([
            StructField("id", StringType(), False),
            StructField("distance", DoubleType(), False)
        ])

        distances_df = self.spark.createDataFrame(distances, schema).filter(col("distance") < float('inf'))
        distances_df.show(15)

        total_distance = distances_df.agg(sql_sum("distance")).collect()[0][0]
        print(f"total_distance = {total_distance}")
        reachable_nodes = distances_df.count()
        print(f"# nodes = {reachable_nodes}")

        # Calcolo della closeness
        if total_distance > 0 and reachable_nodes > 1:
            closeness_centrality = (reachable_nodes - 1) / total_distance
        else:
            closeness_centrality = 0.0

        return (vertex, closeness_centrality)

    def run_analysis(self, file_path):
        """
        Esegue l'intera analisi della rete di retweet, caricando i dati, costruendo il grafo e calcolando
        la centralità di degree e closeness.

        Args:
            file_path (str): Il percorso del file JSON contenente i dati dei tweet.
        """
        users_df, edges_df = self.load_data(file_path)

        # creazione del grafo (non diretto) tramite la libreria graphframe
        g = GraphFrame(users_df, edges_df)

        # Calcolo della centralità di tipo degree
        degree_centrality = g.degrees.cache()
        print("Degree Centrality:")
        degree_centrality.show(15)

        # I primo 15 nodi più importanti secondo la centralità degree
        highest_degree_nodes = degree_centrality.orderBy(col("degree").desc())
        print("Top 10 nodes by Degree Centrality calcolati")
        highest_degree_nodes.show(10)

        n_links_net = edges_df.count()
        print(f"Il numero totale di link nella rete sono {n_links_net}")
        print("Normalizzo i valori della degree per il numero totale di links nella rete e calcolo la percentuale")
        highest_degree_nodes = highest_degree_nodes.withColumn("fract_degree", sql_round(col("degree")/n_links_net * 100, 2))
        highest_degree_nodes.select(col("degree"), col("fract_degree")).show(10)

        # Parte del codice riguardane la closeness, non eseguita a causa delle limitate risorse computazionali
        """
        vertices_ids = [row.id for row in users_df.collect()]
        closeness_centrality = [self.calculate_closeness_centrality(g, v) for v in vertices_ids]

        closeness_schema = StructType([
            StructField("id", StringType(), False),
            StructField("closeness", DoubleType(), False)
        ])

        closeness_centrality_df = self.spark.createDataFrame(closeness_centrality, closeness_schema)
        print("Closeness Centrality:")
        closeness_centrality_df.show(15)
        """

# Percorso al file JSON
file_path = "C:/Users/salvatore/PycharmProjects/progetto_pasquini/.venv/tweets.json"

# Esecuzione dell'analisi
analysis = RetweetNetworkAnalysis(spark)
analysis.run_analysis(file_path)
