# Importazione delle librerie necessarie
import logging
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configurazione del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Classe per il sentiment analysis
class TwitterSentimentAnalysis:
    def __init__(self, dataframe, colonna, stopwords=set()):
        """
         Costruttore: Inizializza la classe con il DataFrame di input, la colonna di testo da analizzare
         e un insieme opzionale di stopwords da filtrare.

         :param dataframe: PySpark DataFrame contenente i dati di input
         :param colonna: stringa, il nome della colonna di testo da analizzare
         :param stopwords: set di stringhe, parole da escludere dall'analisi (default: set vuoto)
        """
        self.dataframe = dataframe
        self.colonna = colonna
        self.stopwords = stopwords
        self.frequenze = {}

    def calcola_frequenze_pyspark(self):
        """
            Calcola le frequenze delle parole nella colonna di testo specificata del DataFrame.

            Questo metodo:
            1. Converte il testo nella colonna specificata in minuscolo.
            2. Divide il testo in parole utilizzando lo spazio come separatore.
            3. Espande le parole in righe separate e filtra le righe vuote.
            4. Filtra le stopwords specificate.
            5. Conta le frequenze delle parole.
            6. Raccoglie i risultati in un dizionario.

            :return: dizionario contenente le parole e le loro frequenze.
        """
        logger.info(f"Calcolo delle frequenze per la colonna: {self.colonna}")
        try:
            # Conversione del testo in minuscolo e suddividi in parole utilizzando lo spazio come separatore
            dataframe = self.dataframe.withColumn(self.colonna, F.lower(F.col(self.colonna)))
            dataframe = dataframe.withColumn(self.colonna, F.split(F.col(self.colonna), "\s+"))

            # Espansione delle parole in righe separate e filtra le righe vuote
            parole_df = dataframe.withColumn("word", F.explode(F.col(self.colonna)))
            parole_df = parole_df.filter(F.col("word") != "")

            # Filtraggio delle stopwords
            parole_df = parole_df.filter(~F.col("word").isin(self.stopwords))

            # Conteggio delle frequenze delle parole
            frequenze_df = parole_df.groupBy("word").count()

            # Raccolta dei risultati in un dizionario
            self.frequenze = {row['word']: row['count'] for row in frequenze_df.collect()}

            logger.info(f"Numero di parole uniche trovate: {len(self.frequenze)}")
            return self.frequenze

        except Exception as e:
            logger.error(f"Errore durante il calcolo delle frequenze: {e}")
            raise

    def traccia_istogramma(self):
        """
            Traccia un istogramma delle frequenze delle parole utilizzando Matplotlib.

            Questo metodo:
            1. Stampa i primi 10 elementi del dizionario delle frequenze per verifica.
            2. Verifica che i valori del dizionario siano numerici.
            3. Ordina le frequenze e seleziona le prime 15 parole più frequenti.
            4. Calcola il numero totale delle frequenze.
            5. Configura e visualizza l'istogramma delle frequenze.
        """
        logger.info("Tracciamento dell'istogramma delle frequenze")
        try:
            frequenze = self.frequenze

            # Stampa e verifica i primi 10 elementi del dizionario
            print("Primi 10 elementi del dizionario:", list(frequenze.items())[:10])

            # Verifica che i valori del dizionario siano numerici
            for k, v in frequenze.items():
                if not isinstance(v, (int, float)):
                    raise TypeError(f"Valore non numerico trovato per la chiave {k}: {v}")

            # Ordinamento delle frequenze e selezione i primi 15
            sorted_frequenze = dict(sorted(frequenze.items(), key=lambda item: item[1], reverse=True)[:15])

            print("Primi 15 elementi ordinati:", sorted_frequenze)

            parole = list(sorted_frequenze.keys())
            valori = list(sorted_frequenze.values())

            # Calcolo del numero totale di parole considerate
            totale_frequenze= sum(sorted_frequenze.values())
            print(f"La somma totale delle frequenze delle parole (dopo aver filtrato le stopwords) è: {totale_frequenze}")

            # Configurazione del grafico
            plt.figure(figsize=(12, 6))
            plt.bar(parole, valori, color='skyblue')
            plt.xlabel('Parole')
            plt.ylabel('Frequenze')
            plt.title('Istogramma delle Frequenze delle Parole')
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('istogramma_frequenze.png')
            plt.show()
        except Exception as e:
            logger.error(f"Errore durante la creazione dell'istogramma: {e}")
            raise

    def crea_wordcloud(self):
        """
          Crea una word cloud delle frequenze delle parole utilizzando la libreria WordCloud.

          Questo metodo:
          1. Genera una word cloud dalle frequenze delle parole.
          2. Configura e visualizza la word cloud.
        """
        logger.info("Creazione della wordcloud delle frequenze")
        try:
            # Generazione della word cloud dalle frequenze
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(self.frequenze)

            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('WordCloud delle Frequenze delle Parole')
            plt.savefig('wordcloud_frequenze.png')
            plt.show()
        except Exception as e:
            logger.error(f"Errore durante la creazione della wordcloud: {e}")
            raise
