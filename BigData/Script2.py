# Importazione delle librerie necessarie
from pyspark import SparkContext, StorageLevel
from pyspark.sql.functions import col, udf, lower, split, explode
from pyspark.sql.types import StringType, BooleanType
from TSA_class import TwitterSentimentAnalysis
from pyspark.sql import SparkSession
import logging
import os, sys
import re
import string

# Configurazione del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Creazione della sessione Spark
spark = SparkSession.builder.getOrCreate()

# Caricamento del data set
data = spark.read.json("C:/Users/salvatore/PycharmProjects/progetto_pasquini/.venv/tweets.json")
# Eliminazione di duplicati eventuali, non controlliamo se ci ci sono righe con la casella corrispondente alla colonna testo mancante, perché
# sappiamo dai controlli negli altri programmi che il testo è presente in tutte le righe del data set preso in considerazione
data_no_dupl = data.dropDuplicates()

usa_tweets = data_no_dupl.filter((col("user.time_zone")).like("%US & Canada%"))

# Selezione dei tweets con gli hashtags
def is_empty_list(lst):
    """
       Scopo: Verificare se una lista è vuota.
       Parametri:
           lst (list): Lista da verificare.
       Restituisce:
           bool: True se la lista è vuota, False altrimenti.
    """
    return len(lst) == 0

is_empty_list_udf = udf(is_empty_list, BooleanType())
data_hash = usa_tweets.withColumn("is_empty", is_empty_list_udf(col("entities.hashtags.text")))

def list_to_string(lst):
    """
        Scopo: Convertire una lista di stringhe in una singola stringa separata da '|'.
        Parametri:
            lst (list): Lista di stringhe da convertire.
        Restituisce:
            str: Stringa contenente gli elementi della lista separati da '|'.
            Se la lista è vuota o None, restituisce una stringa vuota.
     """
    return '' if lst is None or len(lst) == 0 else '|'.join(lst)

list_to_string_udf = udf(list_to_string, StringType())
data_hash_str = data_hash.withColumn("values_hash", list_to_string_udf(col("entities.hashtags.text")))

# Filtraggio dei tweets che menzionano Obama e non Romney
tw_obama = data_hash_str.filter(
    (lower(col("values_hash")).contains("obama") | lower(col("values_hash")).contains("obama2012")) &
    (~lower(col("values_hash")).contains("romney"))
).select(
    col("id"), col("user.name"), col("entities.hashtags.text").alias("hashtags"), col("text")
)

# Filtraggio dei tweets che menzionano Romney e non Obama
tw_romney = data_hash_str.filter(
    (~lower(col("values_hash")).contains("obama")) &
    (lower(col("values_hash")).contains("romney"))
).select(
    col("id"), col("user.name"), col("entities.hashtags.text").alias("hashtags"), col("text")
)

def strip_links(text):
    """
        Scopo: Rimuovere i link dal testo.
        Parametri:
            text (str): Testo da cui rimuovere i link.
        Restituisce:
            str: Testo senza link.
    """
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

def strip_all_entities(text):
    """
        Scopo: Rimuovere entità speciali (@, #) e punteggiatura dal testo.
        Parametri:
            text (str): Testo da cui rimuovere entità speciali e punteggiatura.
        Restituisce:
            str: Testo senza entità speciali e punteggiatura.
     """
    entity_prefixes = ['@','#']
    for separator in string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def pre_process(text):
    """
        Scopo: Preprocessare il testo convertendolo in minuscolo e rimuovendo link ed entità speciali.
        Parametri:
            text (str): Testo da preprocessare.
        Restituisce:
            str: Testo preprocessato.
    """
    text = text.lower()
    text = strip_all_entities(strip_links(text))
    return text

# Creazione di una UDF per la pulizia del testo
pre_process_udf = udf(pre_process, StringType())

logger.info("Defined preprocessing functions and UDF")

obama_clean = tw_obama.withColumn("text_cln", pre_process_udf(col("text")))
logger.info("Cleaned text for OBAMA tweets")

romney_clean = tw_romney.withColumn("text_cln", pre_process_udf(col("text")))
logger.info("Cleaned text for ROMNEY tweets")

# Liste di parole positive e negative
positive_words = ['a+', 'abound', 'abundance', 'abundant', 'accessable', 'accessible', 'acclaim', 'accolades', 'accommodative', 'accomodative', 'accomplish', 'accomplishment', 'accurately', 'achievable', 'achievible', 'adaptable', 'adjustable', 'admirably', 'admiringly', 'adorer', 'adoring', 'adoringly', 'adroitly', 'advantage', 'advantageously', 'advantages', 'adventuresome', 'advocate', 'advocated', 'advocates', 'affable', 'affably', 'affection', 'affinity', 'affirm', 'affirmation', 'affirmative', 'agile', 'agreeably', 'all-around', 'alluringly', 'altruistic', 'amaze', 'amazed', 'amazing', 'ambitious', 'amenable', 'amenity', 'amiabily', 'ample', 'amply', 'amuse', 'amusing', 'amusingly', 'apotheosis', 'appeal', 'appealing', 'appreciate', 'appreciates', 'appreciative', 'appreciatively', 'appropriate', 'approval', 'ardently', 'ardor', 'aspirations', 'assurances', 'assure', 'assuring', 'astonished', 'astonishment', 'astound', 'astounded', 'astounding', 'astoundingly', 'astutely', 'attraction', 'attractive', 'attune', 'audible', 'audibly', 'auspicious', 'authentic', 'authoritative', 'autonomous', 'available', 'avidly', 'awards', 'awed', 'awesome', 'awesomely', 'awesomeness', 'awestruck', 'bargain', 'beautiful', 'beautifullly', 'beautifully', 'beauty', 'beckon', 'beneficent', 'beneficiary', 'benefit', 'benefits', 'benevolent', 'best-selling', 'better-known', 'better-than-expected', 'beutifully', 'blameless', 'bless', 'blessing', 'bliss', 'blissfully', 'blithe', 'blockbuster', 'blossom', 'bonuses', 'boom', 'booming', 'boost', 'bountiful', 'brainiest', 'brainy', 'brand-new', 'breakthrough', 'breakthroughs', 'breathlessness', 'breeze', 'bright', 'brighter', 'brisk', 'bullish', 'buoyant', 'calm', 'calming', 'carefree', 'cashbacks', 'catchy', 'celebrate', 'celebrated', 'celebration', 'celebratory', 'charitable', 'charm', 'charmingly', 'chaste', 'cheapest', 'cheerful', 'cherish', 'civility', 'clarity', 'classy', 'cleanly', 'cohere', 'coherence', 'coherent', 'colorful', 'comfortable', 'comforting', 'commend', 'commendable', 'commodious', 'compact', 'compactly', 'compassionate', 'compatible', 'competitive', 'complement', 'complemented', 'complements', 'compliant', 'comprehensive', 'conciliatory', 'concise', 'confidence', 'confident', 'congratulate', 'congratulation', 'congratulatory', 'consistent', 'contentment', 'contribution', 'convienient', 'convient', 'coolest', 'cooperatively', 'correctly', 'cost-saving', 'courageously', 'courtly', 'cozy', 'credence', 'credible', 'danke', 'danken', 'daringly', 'darling', 'dashing', 'dauntless', 'dawn', 'dazzled', 'dazzling', 'decisive', 'defeat', 'defeated', 'defeating', 'deference', 'delicate', 'delight', 'dependable', 'deservedly', 'desirous', 'destiny', 'detachable', 'devout', 'dexterous', 'dexterously', 'dextrous', 'dignified', 'dignify', 'dignity', 'diligent', 'dirt-cheap', 'distinctive', 'distinguished', 'diversified', 'divine', 'divinely', 'dominated', 'dote', 'doubtless', 'dreamland', 'dumbfounded', 'dummy-proof', 'durable', 'eager', 'earnest', 'earnestly', 'ease', 'eased', 'easier', 'easiest', 'easiness', 'easing', 'easy-to-use', 'ebullience', 'ebulliently', 'ecenomical', 'ecstasy', 'edify', 'educated', 'effective', 'effectively', 'effectiveness', 'effectual', 'efficacious', 'efficient', 'effortlessly', 'effusion', 'effusive', 'effusively', 'effusiveness', 'elan', 'elated', 'elatedly', 'elation', 'electrify', 'elegance', 'elegant', 'elegantly', 'elevate', 'eloquent', 'embolden', 'eminence', 'empower', 'enchanted', 'encouraging', 'encouragingly', 'endearing', 'endorsed', 'endorses', 'energetic', 'energize', 'energy-efficient', 'energy-saving', 'engaging', 'engrossing', 'enhancement', 'enjoy', 'enjoyable', 'enjoying', 'enjoyment', 'enjoys', 'enlightenment', 'enliven', 'ennoble', 'enrapture', 'enraptured', 'enrich', 'enterprising', 'entertain', 'entertaining', 'enthral', 'enthrall', 'enthralled', 'enthuse', 'entice', 'enticing', 'entranced', 'entrancing', 'entrust', 'enviable', 'enviably', 'enviously', 'equitable', 'err-free', 'euphoria', 'euphorically', 'evenly', 'eventful', 'everlasting', 'evocative', 'exaltation', 'exalted', 'exaltedly', 'exalting', 'examplary', 'exceed', 'exceeded', 'exceeding', 'exceedingly', 'exceeds', 'exceled', 'excelled', 'excellent', 'excels', 'exceptionally', 'excited', 'excitedness', 'excites', 'excitingly', 'exellent', 'exemplary', 'exhilarating', 'exhilaratingly', 'exhilaration', 'expansive', 'expeditiously', 'exquisite', 'extraordinary', 'exultant', 'exultation', 'eye-catching', 'eyecatch', 'eyecatching', 'facilitate', 'fair', 'fairly', 'faith', 'fame', 'famed', 'famous', 'fancier', 'fanfare', 'fantastically', 'fascinate', 'fascinating', 'fascinatingly', 'fashionable', 'fashionably', 'fast-growing', 'faster', 'fastest', 'fastest-growing', 'favor', 'favored', 'favour', 'feasibly', 'fecilitous', 'felicitate', 'fertile', 'fervent', 'fervor', 'festive', 'fidelity', 'fiery', 'finest', 'first-in-class', 'flattering', 'flatteringly', 'flawless', 'flexibility', 'flourish', 'flourishing', 'fluent', 'formidable', 'fortitude', 'fortuitously', 'freed', 'freedom', 'fresher', 'ftw', 'fulfillment', 'fun', 'futurestic', 'futuristic', 'gaily', 'gained', 'gainful', 'gaining', 'gallant', 'generous', 'generously', 'genial', 'genius', 'gladly', 'gleeful', 'glimmer', 'glitter', 'glitz', 'glorify', 'gloriously', 'glory', 'glowing', 'glowingly', 'god-send', 'godsend', 'good', 'goodly', 'goodness', 'goodwill', 'gooood', 'gracefully', 'graciously', 'graciousness', 'gratefully', 'gratification', 'gratifies', 'gratify', 'greatness', 'grin', 'groundbreaking', 'guarantee', 'guiltless', 'gumption', 'gush', 'gusto', 'gutsy', 'hale', 'hallmark', 'hallowed', 'handily', 'handsome', 'handsomely', 'happier', 'happiness', 'happy', 'harmless', 'harmoniously', 'harmonize', 'harmony', 'headway', 'heal', 'heartening', 'heartwarming', 'heavenly', 'helpful', 'helping', 'hero', 'heroic', 'high-spirited', 'honest', 'honesty', 'honor', 'honored', 'honoring', 'hooray', 'hopeful', 'hotcakes', 'hottest', 'hug', 'humane', 'humble', 'humor', 'humorously', 'humourous', 'ideal', 'idealize', 'ideally', 'idol', 'idyllic', 'illuminate', 'illuminati', 'ilu', 'imaculate', 'immaculate', 'immense', 'impartial', 'impartiality', 'impeccable', 'impeccably', 'important', 'impressively', 'impressiveness', 'improvement', 'improvements', 'improving', 'incredible', 'indebted', 'indulgence', 'inexpensive', 'ingenious', 'ingeniously', 'ingenuity', 'ingenuously', 'innocuous', 'innovation', 'inpressed', 'insightful', 'insightfully', 'inspire', 'instantly', 'instrumental', 'intelligence', 'intelligible', 'invaluable', 'invigorate', 'inviolable', 'irreproachable', 'irresistible', 'irresistibly', 'issue-free', 'jaw-dropping', 'jollify', 'jolly', 'joyful', 'joyously', 'jubilant', 'jubilation', 'judicious', 'keenly', 'kindliness', 'kindly', 'kindness', 'laud', 'lavish', 'lavishly', 'lawful', 'leads', 'liberate', 'lifesaver', 'likable', 'like', 'liking', 'lionhearted', 'lively', 'logical', 'love', 'loved', 'loveliness', 'lover', 'loves', 'loving', 'low-cost', 'low-priced', 'low-risk', 'lower-priced', 'loyalty', 'lucid', 'lucidly', 'luckiness', 'lucky', 'lucrative', 'lustrous', 'luxuriate', 'luxurious', 'luxury', 'magic', 'magical', 'magnanimous', 'magnificence', 'magnificent', 'magnificently', 'majestic', 'manageable', 'marvel', 'marveled', 'marvelous', 'marvelousness', 'master', 'masterful', 'masters', 'matchless', 'maturely', 'meaningful', 'merciful', 'meritorious', 'merrily', 'merry', 'mesmerize', 'mighty', 'mind-blowing', 'miracle', 'miraculous', 'modern', 'modest', 'modesty', 'monumental', 'motivated', 'neat', 'neatest', 'neatly', 'nicely', 'nifty', 'nimble', 'noble', 'nobly', 'noiseless', 'notably', 'nourishing', 'obsession', 'obsessions', 'obtainable', 'openly', 'openness', 'optimism', 'orderly', 'outdone', 'outperforming', 'outshone', 'outsmart', 'outstandingly', 'outwit', 'ovation', 'overjoyed', 'overtaking', 'overtook', 'painlessly', 'palatial', 'pampered', 'pamperedness', 'panoramic', 'paramount', 'passion', 'patience', 'patient', 'patriot', 'peace', 'peaceable', 'peaceful', 'peerless', 'peppy', 'peps', 'perfection', 'permissible', 'persevere', 'personalized', 'picturesque', 'playfully', 'pleasantly', 'pleased', 'pleases', 'pleasing', 'pleasurably', 'plush', 'plusses', 'poetic', 'poeticize', 'poignant', 'poise', 'polished', 'polite', 'positive', 'positively', 'positives', 'powerfully', 'praiseworthy', 'precious', 'precise', 'preeminent', 'preferable', 'preferes', 'preferring', 'prefers', 'premier', 'prestige', 'priceless', 'privilege', 'prize', 'problem-solver', 'prodigious', 'productive', 'proficient', 'proficiently', 'profusion', 'progressive', 'prominent', 'promise', 'promised', 'promises', 'promoter', 'properly', 'propitiously', 'pros', 'prosperity', 'prosperous', 'proud', 'proven', 'providence', 'prudence', 'prudently', 'punctual', 'pure', 'purposeful', 'qualify', 'quicker', 'quiet', 'radiance', 'rapport', 'rapt', 'raptureous', 'rapturous', 'rapturously', 'rational', 'reaffirmation', 'reasonable', 'reassure', 'receptive', 'recomend', 'recommend', 'recommendation', 'recommended', 'reconciliation', 'recovery', 'rectifying', 'redeem', 'redemption', 'refined', 'reformed', 'refreshed', 'regal', 'regally', 'regard', 'rejoicingly', 'rejuvenate', 'rejuvenated', 'rejuvenating', 'relaxed', 'reliable', 'reliably', 'relish', 'remarkable', 'remedy', 'renewed', 'renown', 'resilient', 'resound', 'resounding', 'resourcefulness', 'respectable', 'resplendent', 'restored', 'restructured', 'restructuring', 'retractable', 'reverent', 'reverently', 'revival', 'revives', 'reward', 'rewardingly', 'richer', 'richness', 'right', 'righten', 'righteousness', 'rightful', 'rightfully', 'rightly', 'rock-star', 'rock-stars', 'rockstars', 'romantic', 'roomier', 'rosy', 'safely', 'sagacity', 'sagely', 'saint', 'salutary', 'salute', 'satisfactorily', 'satisfactory', 'satisfying', 'savings', 'savvy', 'scenic', 'seamless', 'securely', 'self-determination', 'self-respect', 'self-sufficiency', 'sensational', 'sensationally', 'sensations', 'sensible', 'sensitive', 'sexy', 'sharper', 'shine', 'simpler', 'simplest', 'simplified', 'simplifies', 'simplify', 'simplifying', 'sincere', 'sincerely', 'sincerity', 'skillful', 'smartest', 'smoothest', 'snazzy', 'softer', 'solace', 'solid', 'soothe', 'soothingly', 'soulful', 'soundly', 'soundness', 'spacious', 'sparkling', 'spectacular', 'spectacularly', 'spellbind', 'spellbinding', 'spellbindingly', 'spirited', 'spontaneous', 'sporty', 'spotless', 'stabilize', 'standout', 'state-of-the-art', 'statuesque', 'steadfast', 'steadfastly', 'steadfastness', 'steadiest', 'stellar', 'stimulates', 'straighten', 'straightforward', 'streamlined', 'striking', 'strikingly', 'strong', 'stronger', 'strongest', 'stunned', 'stunningly', 'sturdier', 'sturdy', 'suave', 'sublime', 'subsidize', 'subsidizes', 'succeeded', 'succeeding', 'succes', 'success', 'successful', 'sufficed', 'sufficiently', 'suitable', 'sumptuous', 'sumptuously', 'superior', 'superiority', 'supple', 'support', 'supporting', 'supports', 'supremacy', 'surpass', 'sustainability', 'sustainable', 'swank', 'swankier', 'sweeten', 'sweetly', 'sweetness', 'swift', 'swiftness', 'tantalizing', 'tantalizingly', 'tempting', 'temptingly', 'tenacious', 'tenaciously', 'terrific', 'thinner', 'thoughtful', 'thrill', 'thrillingly', 'thrills', 'tidy', 'time-honored', 'timely', 'tingle', 'titillate', 'titillating', 'titillatingly', 'togetherness', 'top-notch', 'top-quality', 'topnotch', 'toughest', 'traction', 'tranquil', 'tranquility', 'transparent', 'triumph', 'trivially', 'trump', 'trumpet', 'trust', 'trusted', 'trusting', 'trustingly', 'trustworthy', 'trusty', 'ultra-crisp', 'unaffected', 'unbiased', 'uncomplicated', 'undaunted', 'undisputably', 'undisputed', 'unencumbered', 'unequivocal', 'unfettered', 'unforgettable', 'unquestionably', 'unreal', 'unrestricted', 'unwavering', 'upgradable', 'uphold', 'uplifting', 'upliftingly', 'upliftment', 'useable', 'valiantly', 'venerate', 'versatile', 'versatility', 'vibrantly', 'viewable', 'vigilant', 'visionary', 'vivid', 'vouch', 'wealthy', 'well-behaved', 'well-connected', 'well-educated', 'well-informed', 'well-known', 'well-made', 'well-mannered', 'well-regarded', 'well-rounded', 'well-run', 'wholesome', 'whooa', 'whoooa', 'willing', 'willingly', 'winnable', 'winners', 'wisdom', 'wisely', 'witty', 'won', 'wonder', 'wonderful', 'wonderfully', 'wonderously', 'wondrous', 'work', 'worked', 'works', 'worth-while', 'worthwhile', 'wow', 'wowing', 'yay', 'zest']

negative_words = ['abnormal', 'abominate', 'abrupt', 'abuse', 'abusive', 'ached', 'acridly', 'adamantly', 'adverse', 'aggravating', 'aggressor', 'agonize', 'alienate', 'allegation', 'allegations', 'ambiguous', 'annoy', 'anomalous', 'antagonism', 'apathetic', 'apologist', 'appalled', 'appalling', 'aspersion', 'audaciousness', 'audacity', 'avalanche', 'ax', 'backaching', 'bad', 'banalize', 'barbaric', 'barbarous', 'barbarously', 'bashing', 'bastards', 'berate', 'beset', 'betrayal', 'bitter', 'blabber', 'blandish', 'blasphemous', 'blasted', 'blather', 'bleed', 'bleeds', 'blur', 'blurs', 'bonkers', 'brazen', 'breach', 'break-ups', 'breakdown', 'brutalising', 'buggy', 'bulky', 'bull----', 'bullyingly', 'burdensomely', 'burn', 'burns', 'buzzing', 'calumnies', 'calumny', 'cancer', 'cancerous', 'cartoonish', 'castrated', 'chatterbox', 'cheesy', 'choke', 'clamor', 'collapse', 'collude', 'collusion', 'concerns', 'condescension', 'confront', 'confused', 'contagious', 'contaminated', 'contaminates', 'contend', 'contrived', 'corrupting', 'counterproductive', 'coward', 'cracks', 'cramped', 'craps', 'cravenly', 'criminal', 'crippled', 'cripples', 'crude', 'curt', 'dark', 'dastard', 'dawdle', 'death', 'debt', 'debts', 'deceitfulness', 'decline', 'decrepitude', 'defamation', 'defensive', 'degenerate', 'degrading', 'delirious', 'demonic', 'derisive', 'despairing', 'despicably', 'despoil', 'despotism', 'deterrent', 'detracted', 'detracting', 'detraction', 'detriment', 'die', 'died', 'dings', 'disadvantage', 'disagree', 'disagreeably', 'disagreed', 'disagrees', 'disapointing', 'disappointingly', 'disappoints', 'disasterous', 'disconcerted', 'discontinuous', 'discouragingly', 'discrepant', 'disdainfully', 'disgust', 'disillusionment', 'disinclination', 'dispiritedly', 'disputable', 'disregard', 'disregardful', 'dissonance', 'distastefully', 'distressingly', 'disvalue', 'dizzy', 'dogged', 'doggedly', 'doomed', 'downbeat', 'downheartedly', 'dragged', 'dripped', 'effigy', 'emphatic', 'entangle', 'equivocal', 'errant', 'errors', 'exacerbate', 'exagerates', 'execrate', 'extermination', 'extinguish', 'faint', 'fallacious', 'fallaciousness', 'fallout', 'faltered', 'famine', 'fanatical', 'farce', 'fatalistically', 'fatique', 'fatuously', 'fearful', 'felon', 'figurehead', 'flakieness', 'flareup', 'flaunt', 'flee', 'flimflam', 'forlornly', 'forswear', 'frantically', 'fright', 'fuck', 'furor', 'gabble', 'gawky', 'ghosting', 'gnawing', 'grainy', 'grate', 'grimace', 'gruff', 'guiltily', 'gutter', 'hacks', 'hardship', 'hastily', 'hatred', 'havoc', 'heartbreaker', 'heavyhearted', 'hegemonistic', 'hideously', 'hoard', 'hubris', 'hurting', 'hysterically', 'idiot', 'illegally', 'imbecile', 'imminently', 'immorality', 'impede', 'imperfections', 'imperfectly', 'imperiously', 'impetuously', 'implode', 'importune', 'impoverished', 'improbability', 'impugn', 'impulsive', 'inarticulate', 'inconsequential', 'indecisive', 'indecisively', 'indeterminate', 'indistinguishable', 'inefficient', 'inept', 'inequalities', 'inescapably', 'inexcusably', 'infection', 'inferior', 'infernal', 'infest', 'infuriate', 'inimically', 'insensitive', 'insulted', 'insurrection', 'intense', 'intimidate', 'intransigence', 'irresolvable', 'isolated', 'issue', 'jealousness', 'killed', 'lack', 'lackadaisical', 'lacked', 'lackeys', 'languid', 'languish', 'languor', 'lapses', 'last-ditch', 'leak', 'licentiously', 'limitations', 'loathsome', 'long-winded', 'longing', 'loses', 'lousy', 'lovelorn', 'ludicrously', 'madly', 'maladjustment', 'malaise', 'maledict', 'malevolently', 'mania', 'manic', 'manipulate', 'mar', 'meager', 'messing', 'misaligned', 'misbecome', 'miscalculate', 'miser', 'miserable', 'misfortune', 'misgiving', 'mispronounce', 'mistake', 'mists', 'misuse', 'mocked', 'mockery', 'moot', 'mordant', 'moron', 'mortification', 'mourner', 'mudslinging', 'nag', 'needy', 'negatives', 'negligence', 'nettle', 'nightmarish', 'nosey', 'obscenity', 'obsessive', 'obstructing', 'odd', 'odder', 'offend', 'offender', 'offensively', 'opponent', 'outburst', 'over-hyped', 'overkill', 'overlook', 'overrated', 'overshadow', 'oversize', 'overthrow', 'overwhelmingly', 'overzelous', 'pain', 'pains', 'panick', 'pedantic', 'perplexity', 'persecute', 'persecution', 'pervasive', 'petrified', 'pittance', 'plaything', 'plight', 'poison', 'polemize', 'poorer', 'prison', 'prosecute', 'protest', 'protracted', 'punitive', 'puzzlement', 'quitter', 'racism', 'radical', 'rail', 'rampant', 'ramshackle', 'redundant', 'regrettably', 'rejects', 'reluctantly', 'repetitive', 'reprehension', 'reproach', 'repugnance', 'repulsed', 'repulsiveness', 'restrict', 'restricted', 'restrictive', 'reticent', 'retreated', 'revengeful', 'rile', 'rip', 'ruins', 'ruthlessness', 'sabotage', 'sacrificed', 'sadly', 'scams', 'scandalize', 'scandals', 'scarce', 'scathing', 'screwed-up', 'second-class', 'self-coup', 'self-destructive', 'selfishly', 'shaky', 'shameful', 'shamelessly', 'shamelessness', 'short-lived', 'shun', 'sicken', 'sin', 'skittishly', 'slap', 'sloth', 'smash', 'smelling', 'smouldering', 'smudging', 'soapy', 'spank', 'spinster', 'spiritless', 'spooky', 'spurious', 'stern', 'stigma', 'stolen', 'stormy', 'stress', 'stressfully', 'stricken', 'stumped', 'stunt', 'stupid', 'stupidest', 'stupify', 'stutter', 'subordinate', 'subvert', 'sueing', 'sugar-coat', 'sugar-coated', 'suppression', 'swindle', 'swollen', 'symptoms', 'tarnishing', 'temptation', 'tenuous', 'testy', 'thankless', 'thicker', 'throb', 'tingled', 'tingling', 'tired', 'top-heavy', 'topple', 'tramp', 'treachery', 'tricky', 'uglier', 'ultra-hardline', 'unbearablely', 'unbelievably', 'unclear', 'uncollectible', 'uncompromisingly', 'uncontrolled', 'unconvincingly', 'undercuts', 'undercutting', 'undermining', 'unfamiliar', 'unfriendly', 'unhealthy', 'unimaginably', 'uninsured', 'unjust', 'unlamentably', 'unlawfulness', 'unproved', 'unreliability', 'unsavory', 'unspeakable', 'unspecified', 'unsuccessfully', 'untouched', 'unuseable', 'unuseably', 'unwise', 'uproarously', 'upsets', 'usurp', 'vain', 'vehemently', 'vibrated', 'vibrates', 'wail', 'wallow', 'warped', 'weaker', 'weaknesses', 'weariness', 'whiny', 'worried', 'wrath']

logger.info("Defined lists of positive and negative words")

def calculate_sentiment(text):
    """
        Scopo: Calcolare il sentimento del testo basato su parole positive e negative.
        Parametri:
            text (str): Testo su cui calcolare il sentimento.
        Restituisce:
            str: Sentimento del testo ('positivo', 'negativo' o 'neutro').
    """
    words = text.split()
    pos_count = 0
    neg_count = 0
    for word in words:
        if word in positive_words:
            pos_count += 1
        if word in negative_words:
            neg_count += 1
    if pos_count > neg_count:
        return "positivo"
    elif neg_count > pos_count:
        return "negativo"
    else:
        return "neutro"

calculate_sentiment_udf = udf(calculate_sentiment, StringType())
logger.info("Registered UDF for sentiment calculation")

obama_sent = obama_clean.withColumn("sentiment", calculate_sentiment_udf(col("text_cln")))
romney_sent = romney_clean.withColumn("sentiment", calculate_sentiment_udf(col("text_cln")))
logger.info("Applied sentiment UDF to OBAMA and ROMNEY dataframes")

# Filtraggio dei DataFrame per sentiment
oba_pos = obama_sent.filter(col("sentiment") == "positivo")
oba_neg = obama_sent.filter(col("sentiment") == "negativo")
rom_pos = romney_sent.filter(col("sentiment") == "positivo")
rom_neg = romney_sent.filter(col("sentiment") == "negativo")
logger.info("Sentiment selection done")

# Definizione di uno stopwords set
stopwords = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could',
    'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s',
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m',
    'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t',
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves',
    'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such',
    'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they',
    'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
    'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'with', 'won\'t', 'would', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your',
    'yours', 'yourself', 'yourselves', 'us', 'rt', 'p', 'dont', 'will', '...', '-', 's', 't', 'u', '4', '11', 'm', '47', '50',
    '&amp;', '&amp', 'amp', '9'
])

# Istanziamento degli oggetti di tipo: TwitterSentimentAnalysis

col_1 = TwitterSentimentAnalysis(tw_obama, "text", stopwords)
frequenze = col_1.calcola_frequenze_pyspark()
col_1.traccia_istogramma()
col_1.crea_wordcloud()

col_2 = TwitterSentimentAnalysis(tw_romney, "text", stopwords)
frequenze = col_2.calcola_frequenze_pyspark()
col_2.traccia_istogramma()
col_2.crea_wordcloud()

col_3 = TwitterSentimentAnalysis(obama_clean, "text_cln", stopwords)
frequenze = col_3.calcola_frequenze_pyspark()
col_3.traccia_istogramma()
col_3.crea_wordcloud()

col_4 = TwitterSentimentAnalysis(romney_clean, "text_cln", stopwords)
frequenze = col_4.calcola_frequenze_pyspark()
col_4.traccia_istogramma()
col_4.crea_wordcloud()

col_5 = TwitterSentimentAnalysis(oba_pos, "text_cln", stopwords)
frequenze = col_5.calcola_frequenze_pyspark()
col_5.traccia_istogramma()
col_5.crea_wordcloud()

col_6 = TwitterSentimentAnalysis(rom_neg, "text_cln", stopwords)
frequenze = col_6.calcola_frequenze_pyspark()
col_6.traccia_istogramma()
col_6.crea_wordcloud()

col_7 = TwitterSentimentAnalysis(rom_pos, "text_cln", stopwords)
frequenze = col_7.calcola_frequenze_pyspark()
col_7.traccia_istogramma()
col_7.crea_wordcloud()

col_8 = TwitterSentimentAnalysis(oba_neg, "text_cln", stopwords)
frequenze = col_8.calcola_frequenze_pyspark()
col_8.traccia_istogramma()
col_8.crea_wordcloud()
