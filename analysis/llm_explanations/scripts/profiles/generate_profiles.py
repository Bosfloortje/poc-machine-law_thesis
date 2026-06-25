#!/usr/bin/env python3
"""
Generate new profiles based on existing profiles.yaml patterns.

This generator uses rule-based profile generation - NO LLM required.
It analyzes patterns from existing profiles and creates realistic variations.

Usage:
    python scripts/generators/generate_profiles.py --count 100 --output new_profiles.yaml
"""

import argparse
import random
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy


# =============================================================================
# DIVERSE NAMEN - Eerlijke verdeling van achtergronden
# =============================================================================

# Nederlandse voornamen (traditioneel + modern)
FIRST_NAMES_M_NL = [
    "Jan", "Piet", "Henk", "Willem", "Gerrit", "Dirk", "Johan", "Cornelis",
    "Daan", "Lars", "Thijs", "Bram", "Ruben", "Tim", "Sven", "Finn", "Luuk", "Stijn",
    "Thomas", "Bas", "Joost", "Maarten", "Jeroen", "Pieter", "Wouter", "Michiel"
]

FIRST_NAMES_F_NL = [
    "Anna", "Maria", "Elisabeth", "Johanna", "Cornelia", "Hendrika",
    "Sophie", "Emma", "Julia", "Lisa", "Sara", "Eva", "Laura", "Lotte", "Fleur",
    "Iris", "Sanne", "Marieke", "Femke", "Anouk", "Nienke", "Roos", "Maud"
]

# Marokkaanse/Arabische namen
FIRST_NAMES_M_AR = [
    "Mohammed", "Ahmed", "Youssef", "Omar", "Ali", "Hassan", "Ibrahim", "Karim",
    "Rachid", "Samir", "Jamal", "Khalid", "Bilal", "Tariq", "Mustafa", "Nabil"
]

FIRST_NAMES_F_AR = [
    "Fatima", "Amira", "Aya", "Layla", "Nadia", "Samira", "Yasmin", "Karima",
    "Zahra", "Souad", "Malika", "Farida", "Hafsa", "Khadija", "Salma", "Nour"
]

# Turkse namen
FIRST_NAMES_M_TR = [
    "Mehmet", "Mustafa", "Ahmet", "Emre", "Burak", "Ozan", "Serkan", "Kemal",
    "Yusuf", "Hakan", "Murat", "Cem", "Deniz", "Berk", "Kaan", "Umut"
]

FIRST_NAMES_F_TR = [
    "Ayse", "Fatma", "Elif", "Zeynep", "Merve", "Esra", "Derya", "Selin",
    "Ebru", "Gul", "Canan", "Sibel", "Asli", "Tugba", "Yasemin", "Pinar"
]

# Surinaamse/Caribische namen
FIRST_NAMES_M_SR = [
    "Rajen", "Sunil", "Ashwin", "Ravi", "Chandresh", "Praveen", "Rajesh", "Vinod",
    "Clifton", "Rodney", "Dwight", "Marlon", "Humphrey", "Iwan", "Sergio", "Glenn"
]

FIRST_NAMES_F_SR = [
    "Sharda", "Anita", "Radha", "Kamla", "Sunita", "Nalini", "Geeta", "Indira",
    "Shirley", "Gladys", "Ingrid", "Miriam", "Gwendolyn", "Jeanette", "Esther", "Naomi"
]

# Chinese/Aziatische namen
FIRST_NAMES_M_AS = [
    "Wei", "Ming", "Chen", "Jian", "Hao", "Long", "Feng", "Tao",
    "Kenji", "Hiroshi", "Takeshi", "Yuki", "Kim", "Sang", "Min", "Hyun"
]

FIRST_NAMES_F_AS = [
    "Mei", "Li", "Xiu", "Ying", "Fang", "Jing", "Hui", "Yan",
    "Yuki", "Sakura", "Hana", "Akiko", "Ji-young", "Soo-yeon", "Min-ji", "Hye-jin"
]

# Oost-Europese namen
FIRST_NAMES_M_EE = [
    "Piotr", "Krzysztof", "Andrzej", "Tomasz", "Marek", "Pawel", "Jan", "Adam",
    "Ivan", "Dmitri", "Sergei", "Vladimir", "Alexandru", "Mihai", "Stefan", "Bogdan"
]

FIRST_NAMES_F_EE = [
    "Anna", "Katarzyna", "Malgorzata", "Agnieszka", "Ewa", "Monika", "Joanna", "Beata",
    "Natasha", "Olga", "Elena", "Tatiana", "Maria", "Ioana", "Andreea", "Cristina"
]

# Combineer alle namen met gelijke gewichten per groep
FIRST_NAMES_M = {
    'NL': FIRST_NAMES_M_NL,
    'AR': FIRST_NAMES_M_AR,
    'TR': FIRST_NAMES_M_TR,
    'SR': FIRST_NAMES_M_SR,
    'AS': FIRST_NAMES_M_AS,
    'EE': FIRST_NAMES_M_EE,
}

FIRST_NAMES_F = {
    'NL': FIRST_NAMES_F_NL,
    'AR': FIRST_NAMES_F_AR,
    'TR': FIRST_NAMES_F_TR,
    'SR': FIRST_NAMES_F_SR,
    'AS': FIRST_NAMES_F_AS,
    'EE': FIRST_NAMES_F_EE,
}

# Achternamen per achtergrond
LAST_NAMES_NL = [
    "de Jong", "Jansen", "de Vries", "van den Berg", "van Dijk", "Bakker", "Janssen",
    "Visser", "Smit", "Meijer", "de Boer", "Mulder", "de Groot", "Bos", "Vos",
    "Peters", "Hendriks", "van Leeuwen", "Dekker", "Brouwer", "de Wit", "Dijkstra"
]

LAST_NAMES_AR = [
    "El-Amrani", "Benali", "El-Idrissi", "Bouazza", "El-Haddaoui", "Amrani", "Chakir",
    "El-Moussaoui", "Bouziane", "El-Alaoui", "Tahiri", "Ziani", "Berrada", "Kadiri"
]

LAST_NAMES_TR = [
    "Yilmaz", "Kaya", "Demir", "Celik", "Sahin", "Yildiz", "Ozturk", "Aydin",
    "Arslan", "Dogan", "Kilic", "Aslan", "Erdogan", "Polat", "Ozkan", "Koc"
]

LAST_NAMES_SR = [
    "Ramdin", "Persaud", "Ramdas", "Jagessar", "Baldewsingh", "Soekhlal", "Raghoebar",
    "Tjon-A-Loi", "Wijngaarde", "Redan", "Amelo", "Sedoc", "Kromo", "Biharie"
]

LAST_NAMES_AS = [
    "Wang", "Li", "Zhang", "Chen", "Liu", "Yang", "Huang", "Wu",
    "Tanaka", "Suzuki", "Kim", "Park", "Lee", "Nguyen", "Tran", "Pham"
]

LAST_NAMES_EE = [
    "Kowalski", "Nowak", "Wisniewski", "Wojcik", "Kozlowski", "Kaminski",
    "Ivanov", "Petrov", "Sidorov", "Popov", "Ionescu", "Popa", "Popescu", "Stoica"
]

LAST_NAMES = {
    'NL': LAST_NAMES_NL,
    'AR': LAST_NAMES_AR,
    'TR': LAST_NAMES_TR,
    'SR': LAST_NAMES_SR,
    'AS': LAST_NAMES_AS,
    'EE': LAST_NAMES_EE,
}

# CBS herkomstverdeling bevolking 2023 (Bevolkingsstatistiek CBS)
# 75.1% autochtoon (beide ouders NL-geboren) + 6.4% één buitenlandse ouder → ~81.5% NL naam
# 5.4% tweede generatie niet-westers (voornamelijk Marokkaans en Turks)
# 16.8% buitenlands geboren (verdeeld over diverse herkomstlanden)
BACKGROUND_WEIGHTS = {
    'NL': 81.5,  # autochtoon + één buitenlandse ouder
    'AR': 6.0,   # Marokkaans (2e gen + buitenlands geboren)
    'TR': 5.5,   # Turks
    'SR': 4.0,   # Surinaams/Antilliaans
    'AS': 2.0,   # Aziatisch/overig niet-westers
    'EE': 1.0,   # Oost-Europees/overig westers
}
_BG_KEYS = list(BACKGROUND_WEIGHTS.keys())
_BG_PROBS = [BACKGROUND_WEIGHTS[k] / sum(BACKGROUND_WEIGHTS.values()) for k in _BG_KEYS]

# CBS leeftijdsverdeling volwassen bevolking 2023
# Bron: CBS Bevolkingsstatistiek (% van totale bevolking incl. kinderen 0-17)
#   18-19: 2.1%, 20-39: 26%, 40-64: 32%, 65-79: 16%, 80+: 5%  (sommeren naar ~81.1%)
# De overige ~18.9% zijn kinderen — buiten scope van onze profielen (18+).
# random.choices normaliseert de gewichten automatisch; effectieve 18+-verdeling:
#   18-19: 2.6%, 20-39: 32.1%, 40-64: 39.5%, 65-79: 19.7%, 80+: 6.2%
AGE_GROUPS = [
    (18, 19),   # Jongvolwassenen
    (20, 39),   # Jonge werkenden en gezinsvorming
    (40, 64),   # Middenleeftijd, piek arbeidsparticipatie
    (65, 79),   # Vroegpensioen en AOW
    (80, 90),   # Hoogbejaarden
]
AGE_GROUP_WEIGHTS = [2.1, 26.0, 32.0, 16.0, 5.0]  # CBS % van totale bevolking; random.choices normaliseert naar 18+-aandeel

# Per-werkstatus leeftijdsverdeling — voorkomt dat gepensioneerden leeftijd 35 krijgen
# Werkenden (loondienst + werkloosheid): alleen 18-64
# Genormaliseerd uit CBS (18-64 = 2.1+26+32 = 60.1): 3.5%, 43.3%, 53.2%
WERKEND_AGE_GROUPS  = [(18, 19), (20, 39), (40, 64)]
WERKEND_AGE_WEIGHTS = [3.5, 43.3, 53.2]

# Gepensioneerden: alleen 65+
# Genormaliseerd uit CBS: 16+5 = 21 → 76.2%, 23.8%
GEPENSIONEERD_AGE_GROUPS  = [(65, 79), (80, 90)]
GEPENSIONEERD_AGE_WEIGHTS = [76.2, 23.8]

# ZZP-specifieke leeftijdsverdeling (CBS zzp-statistieken 2023, 1,2M zzp'ers)
ZZP_AGE_GROUPS = [(18, 24), (25, 44), (45, 74)]
ZZP_AGE_WEIGHTS  = [4.9, 35.1, 60.0]   # CBS: 15-25: 4.9%, 25-45: 35.1%, 45-75: 60%
ZZP_GENDER_WEIGHTS = [62.0, 38.0]      # CBS: 62% man, 38% vrouw

# CBS gemiddeld bruto-inkomen per leeftijdsklasse (2023, in eurocenten)
# Bron: CBS Inkomen van personen naar leeftijd
CBS_INKOMEN_PER_LEEFTIJD = [
    ((18, 24),  1_780_000),   # €17.800
    ((25, 34),  5_260_000),   # €52.600
    ((35, 44),  6_410_000),   # €64.100
    ((45, 54),  6_820_000),   # €68.200
    ((55, 64),  6_150_000),   # €61.500
    ((65, 74),  2_810_000),   # €28.100
    ((75, 84),  1_280_000),   # €12.800
    ((85, 99),  1_430_000),   # €14.300
]

# CBS werkstatus verdeling 18+ bevolking (2023)
# Bron: CBS Arbeidsmarkt in cijfers 2023
#   Werkzamen:     9.8M totaal (vast 5.6M + flex 2.7M = 8.3M employee, zzp 1.2M)
#   Werkloos:      ~4% van beroepsbevolking ≈ 0.41M
#   Gepensioneerd: 20.8% van totale bevolking (~17.9M) ≈ 3.72M
# Normalisatie over de 4 categorieën: 13.93M totaal
WORK_STATUS_WEIGHTS = {
    'employee':      59.6,   # 8.3M / 13.93M — vast + flex loondienst
    'gepensioneerd': 26.7,   # 3.72M / 13.93M
    'zzp':            8.6,   # 1.2M / 13.93M
    'werkloosheid':   2.9,   # 0.41M / 13.93M
}
_WS_KEYS  = list(WORK_STATUS_WEIGHTS.keys())
_WS_PROBS = [WORK_STATUS_WEIGHTS[k] / sum(WORK_STATUS_WEIGHTS.values()) for k in _WS_KEYS]
WORK_STATUSES = _WS_KEYS  # backwards-compat alias

# Inkomensniveaus voor eerlijke verdeling (3 categorieën = ~33% elk)
INCOME_LEVELS = ['laag', 'midden', 'hoog']

# Gemeenten met postcodes
GEMEENTEN = {
    "Amsterdam": {"postcodes": range(1000, 1110), "arrondissement": "AMSTERDAM", "rechtbank": "RECHTBANK_AMSTERDAM"},
    "Rotterdam": {"postcodes": range(3000, 3100), "arrondissement": "ROTTERDAM", "rechtbank": "RECHTBANK_ROTTERDAM"},
    "Den Haag": {"postcodes": range(2500, 2600), "arrondissement": "DEN_HAAG", "rechtbank": "RECHTBANK_DEN_HAAG"},
    "Utrecht": {"postcodes": range(3500, 3600), "arrondissement": "MIDDEN-NEDERLAND", "rechtbank": "RECHTBANK_MIDDEN_NEDERLAND"},
    "Eindhoven": {"postcodes": range(5600, 5660), "arrondissement": "OOST_BRABANT", "rechtbank": "RECHTBANK_OOST_BRABANT"},
    "Groningen": {"postcodes": range(9700, 9750), "arrondissement": "NOORD-NEDERLAND", "rechtbank": "RECHTBANK_NOORD_NEDERLAND"},
    "Maastricht": {"postcodes": range(6200, 6230), "arrondissement": "LIMBURG", "rechtbank": "RECHTBANK_LIMBURG"},
}

STRATEN = [
    "Meeuwenlaan", "Amsteldijk", "Kerkstraat", "Dorpsstraat", "Hoofdweg", "Schoolstraat",
    "Parkweg", "Molenweg", "Beatrixlaan", "Wilhelminastraat", "Nassaulaan"
]


def get_cbs_income(age: int, income_level: str) -> int:
    """Return yearly income in eurocenten, gebaseerd op CBS-gemiddelde voor leeftijdsklasse.

    'laag'  → 35–65% van het CBS-gemiddelde voor die leeftijdsgroep
    'midden' → 85–115% van het CBS-gemiddelde
    'hoog'  → 150–250% van het CBS-gemiddelde
    Minimum €8.000 (800_000 cent) om onrealistische nullen te voorkomen.
    """
    gemiddeld = 3_500_000  # fallback €35.000
    for (min_age, max_age), avg in CBS_INKOMEN_PER_LEEFTIJD:
        if min_age <= age <= max_age:
            gemiddeld = avg
            break

    if income_level == 'laag':
        factor = random.uniform(0.35, 0.65)
    elif income_level == 'hoog':
        factor = random.uniform(1.50, 2.50)
    else:  # midden
        factor = random.uniform(0.85, 1.15)

    return max(800_000, int(gemiddeld * factor))


def load_existing_profiles(yaml_path):
    """Load existing profiles to learn patterns."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data['globalServices'], data['profiles']


def generate_bsn(used_bsns):
    """Generate unique BSN number."""
    while True:
        bsn = str(random.randint(100000000, 999999999))
        if bsn not in used_bsns:
            used_bsns.add(bsn)
            return bsn


def generate_geboortedatum(min_age=18, max_age=80, age_group=None):
    """Generate random birth date, optionally within specific age group."""
    if age_group:
        age = random.randint(age_group[0], age_group[1])
    else:
        age = random.randint(min_age, max_age)
    birth_year = datetime.now().year - age
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Safe day range
    return f"{birth_year}-{month:02d}-{day:02d}", age


def get_name_for_background(background, is_male):
    """Get appropriate first and last name for a background."""
    if is_male:
        first_name = random.choice(FIRST_NAMES_M[background])
    else:
        first_name = random.choice(FIRST_NAMES_F[background])
    last_name = random.choice(LAST_NAMES[background])
    return first_name, last_name


def generate_address(gemeente):
    """Generate random address."""
    info = GEMEENTEN[gemeente]
    straat = random.choice(STRATEN)
    huisnummer = str(random.randint(1, 200))
    postcode_num = random.choice(list(info['postcodes']))
    postcode_letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
    postcode = f"{postcode_num}{postcode_letters}"

    return {
        'straat': straat,
        'huisnummer': huisnummer,
        'postcode': postcode,
        'woonplaats': gemeente,
        'full_address': f"{straat} {huisnummer}, {postcode} {gemeente}",
        'arrondissement': info['arrondissement'],
        'rechtbank': info['rechtbank']
    }


def generate_description(profile_data):
    """Generate profile description based on data."""
    descriptions = []

    # Work status
    if profile_data.get('is_zzper'):
        descriptions.append("ZZP'er")
    elif profile_data.get('werkloosheid'):
        if profile_data.get('bijstand_doelgroep'):
            descriptions.append("Werkloos (bijstand-doelgroep, geen inkomen)")
        else:
            descriptions.append("Werkloos")
    elif profile_data.get('is_employee'):
        descriptions.append("In loondienst")
    elif profile_data.get('is_gepensioneerd'):
        descriptions.append("Gepensioneerd")

    # Income
    if profile_data.get('laag_inkomen'):
        descriptions.append("met laag inkomen")
    elif profile_data.get('hoog_inkomen'):
        descriptions.append("met hoog inkomen")

    # Family
    if profile_data.get('has_partner'):
        descriptions.append("met partner")
    else:
        descriptions.append("alleenstaand")

    if profile_data.get('has_children'):
        num_children = profile_data.get('num_children', 1)
        if num_children == 1:
            descriptions.append("met 1 kind")
        else:
            descriptions.append(f"met {num_children} kinderen")

        if profile_data.get('child_with_condition'):
            descriptions.append("waarvan één met chronische aandoening")

    return ", ".join(descriptions) + "."


# =============================================================================
# DATA SOURCE GENERATORS
# =============================================================================

def generate_belastingdienst(bsn, profile_data):
    """Generate Belastingdienst data based on work status and income level."""
    is_zzper = profile_data.get('is_zzper', False)
    is_employee = profile_data.get('is_employee', False)
    is_gepensioneerd = profile_data.get('is_gepensioneerd', False)
    werkloosheid = profile_data.get('werkloosheid', False)
    laag_inkomen = profile_data.get('laag_inkomen', False)
    hoog_inkomen = profile_data.get('hoog_inkomen', False)

    # Gebruik het CBS-gewogen inkomen dat in generate_profile is berekend
    yearly_income = profile_data.get('yearly_income', 3_500_000)
    if laag_inkomen:
        savings = random.randint(0, 500_000)           # 0–5k
    elif hoog_inkomen:
        savings = random.randint(2_000_000, 10_000_000)  # 20k–100k
    else:
        savings = random.randint(500_000, 2_000_000)   # 5k–20k

    monthly_income = yearly_income // 12

    # Default values
    loon = 0
    uitkeringen = 0
    winst_onderneming = 0
    business_income = 0

    if is_zzper:
        winst_onderneming = yearly_income
        business_income = yearly_income
    elif is_employee:
        loon = yearly_income
    elif is_gepensioneerd:
        uitkeringen = yearly_income
    elif werkloosheid:
        uitkeringen = yearly_income

    return {
        'box1': [{
            'bsn': bsn,
            'loon_uit_dienstbetrekking': loon,
            'uitkeringen_en_pensioenen': uitkeringen,
            'winst_uit_onderneming': winst_onderneming,
            'resultaat_overige_werkzaamheden': 0,
            'eigen_woning': random.choice([0, -50000, -85000]) if not laag_inkomen else 0
        }],
        'box2': [{
            'bsn': bsn,
            'reguliere_voordelen': 0,
            'vervreemdingsvoordelen': 0
        }],
        'box3': [{
            'bsn': bsn,
            'spaargeld': savings,
            'beleggingen': random.randint(0, savings // 2) if hoog_inkomen else 0,
            'onroerend_goed': 0,
            'schulden': 0,
            'buitenlands_inkomen': 0
        }],
        'monthly_income': [{'bsn': bsn, 'bedrag': monthly_income}],
        'maandelijks_inkomen': [{'bsn': bsn, 'bedrag': monthly_income}],
        'assets': [{'bsn': bsn, 'bedrag': savings}],
        'bezittingen': [{'bsn': bsn, 'bedrag': savings}],
        'business_income': [{'bsn': bsn, 'bedrag': business_income}],
        'bedrijfsinkomen': [{'bsn': bsn, 'bedrag': business_income}],
        'buitenlands_inkomen': [{'bsn': bsn, 'bedrag': 0, 'land': 'GEEN'}],
        'aftrekposten': [{'bsn': bsn, 'persoonsgebonden_aftrek': 0}],
        'belastingdienst_vermogen': [{'bsn': bsn, 'vermogen': savings}]
    }


# Activiteiten waarvoor een Alcoholwetvergunning vereist is (Alcoholwet art. 1)
HORECA_ACTIVITEITEN = {'Horeca', 'Restaurant', 'Cafe', 'Catering', 'Eetcafe', 'Lunchroom', 'Snackbar'}
SLIJTERIJ_ACTIVITEITEN = {'Slijterij'}


def generate_kvk(bsn, profile_data, full_name):
    """Generate KVK data for ZZP'ers, incl. alcoholwet data for horeca."""
    if not profile_data.get('is_zzper', False):
        return None

    kvk_nummer = f"{random.randint(10000000, 99999999)}"
    last_name = full_name.split()[-1]
    handelsnaam = f"{last_name} Services"
    # Gewogen activiteitenlijst — horeca/slijterij krijgt ~30% kans zodat er
    # voldoende profielen zijn om alcoholwet-regels mee te testen.
    activiteiten = [
        # Dienstverlening (~25%)
        'Consultancy', 'IT-diensten', 'Coaching', 'Marketing', 'Administratie',
        'Vertaling', 'Training', 'Design', 'Fotografie',
        # Zorg (~10%)
        'Thuiszorg', 'Fysiotherapie', 'Tandartspraktijk', 'Psychologiepraktijk',
        # Horeca (~22%) — hoger gewicht voor alcoholwet-testdekking
        'Horeca', 'Restaurant', 'Cafe', 'Catering',
        'Horeca', 'Restaurant', 'Cafe',  # extra gewicht
        'Eetcafe', 'Lunchroom', 'Snackbar',
        # Slijterij (~8%) — apart type_bedrijf onder Alcoholwet
        'Slijterij', 'Slijterij', 'Slijterij',
        # Voeding overig
        'Bakkerij', 'Slagerij',
        # Bouw & techniek (~12%)
        'Bouw', 'Schilderwerk', 'Loodgieter', 'Elektricien', 'Timmerman', 'Metselaar',
        # Persoonlijke verzorging (~6%)
        'Kapperszaak', 'Schoonheidssalon', 'Nagelsalon',
        # Transport (~6%)
        'Taxivervoer', 'Bezorgdienst', 'Transport',
        # Retail & overig (~8%)
        'Webshop', 'Bloemist', 'Autohandel', 'Schoonmaakbedrijf',
    ]

    activiteit = random.choice(activiteiten)
    is_horeca = activiteit in HORECA_ACTIVITEITEN or activiteit in SLIJTERIJ_ACTIVITEITEN

    if is_horeca:
        horeca_namen = [
            f'Cafe {last_name}', f'Restaurant {last_name}', f"{last_name}'s Eetcafe",
            f'De {last_name}', f'Brasserie {last_name}', f'{last_name} Horeca',
        ]
        handelsnaam = random.choice(horeca_namen)

    kvk_data = {
        'organisaties': [{
            'kvk_nummer': kvk_nummer,
            'handelsnaam': handelsnaam,
            'rechtsvorm': 'EENMANSZAAK',
            'status': 'Actief',
            'aantal_werknemers': random.randint(1, 10),
            'datum_telling': '2024-01-01',
            'datum_aanvang': f"{random.randint(2005, 2022)}-{random.randint(1,12):02d}-01",
            'vestigingsadres': profile_data.get('gemeente', 'Amsterdam'),
        }],
        'inschrijvingen': [{
            'bsn': bsn,
            'kvk_nummer': kvk_nummer,
            'handelsnaam': handelsnaam,
            'rechtsvorm': 'EENMANSZAAK',
            'status': 'ACTIEF',
            'activiteit': activiteit,
        }],
        'functionarissen': [{
            'bsn': bsn,
            'kvk_nummer': kvk_nummer,
            'handelsnaam': handelsnaam,
            'rechtsvorm': 'EENMANSZAAK',
            'status': 'ACTIEF',
            'functie': 'EIGENAAR',
            'bevoegdheid': 'VOLLEDIG',
        }],
        'is_entrepreneur': [{'bsn': bsn, 'waarde': True}],
    }

    if is_horeca:
        type_bedrijf = 'slijtersbedrijf' if activiteit in SLIJTERIJ_ACTIVITEITEN else 'horecabedrijf'
        vloeroppervlakte = random.randint(30, 250)  # art. 10 eist min. 35m2

        # Vergunningscenario's — gelijkmatig verdeeld voor testdekking:
        #   VERLEEND      (~40%): heeft vergunning, compliant
        #   AANGEVRAAGD   (~20%): aanvraag loopt, nog geen vergunning
        #   GEWEIGERD     (~15%): aanvraag afgewezen (bijv. Bibob-bezwaar)
        #   INGETROKKEN   (~10%): vergunning was er maar is ingetrokken
        #   GEEN          (~15%): nooit aangevraagd, opereert zonder
        vergunning_status = random.choices(
            ['VERLEEND', 'AANGEVRAAGD', 'GEWEIGERD', 'INGETROKKEN', 'GEEN'],
            weights=[40, 20, 15, 10, 15],
        )[0]
        heeft_alcoholvergunning = vergunning_status == 'VERLEEND'
        heeft_exploitatievergunning = vergunning_status in ('VERLEEND', 'AANGEVRAAGD')

        kvk_data['inschrijvingen'][0]['type_bedrijf'] = type_bedrijf
        kvk_data['inschrijvingen'][0]['vloeroppervlakte_horecalokaliteit'] = vloeroppervlakte

        # Leidinggevenden — de ondernemer zelf
        kvk_data['leidinggevenden'] = [{
            'kvk_nummer': kvk_nummer,
            'bsn': bsn,
            'naam': full_name,
            'functie': 'LEIDINGGEVENDE',
        }]

        # Fysieke locatie (Alcoholwet art. 10)
        kvk_data['inrichtingen'] = [{
            'kvk_nummer': kvk_nummer,
            'handelsnaam': handelsnaam,
            'type_bedrijf': type_bedrijf,
            'vloeroppervlakte_horecalokaliteit': vloeroppervlakte,
        }]

        # Vergunningenstatus
        kvk_data['vergunningen'] = [{
            'kvk_nummer': kvk_nummer,
            'vergunning_status': vergunning_status,
            'heeft_alcoholvergunning': heeft_alcoholvergunning,
            'heeft_exploitatievergunning': heeft_exploitatievergunning,
        }]

        # Sla op voor SVH/LBB generators
        profile_data['is_horeca'] = True
        profile_data['kvk_nummer'] = kvk_nummer
        profile_data['heeft_alcoholvergunning'] = heeft_alcoholvergunning

    return kvk_data


def generate_svh(bsn, profile_data):
    """Generate SVH Register Sociale Hygiene data (Alcoholwet art. 8 lid 4).
    Leidinggevenden moeten ingeschreven zijn in het Register Sociale Hygiene.
    """
    if not profile_data.get('is_zzper', False):
        return None

    is_horeca = profile_data.get('is_horeca', False)

    if is_horeca:
        # Kans op registratie hangt af van vergunningsstatus:
        # compliant (VERLEEND) → hoge kans; niet-compliant → lagere kans
        heeft_vergunning = profile_data.get('heeft_alcoholvergunning', False)
        kans = 0.90 if heeft_vergunning else 0.45
        is_geregistreerd = random.random() < kans
    else:
        # Niet-horeca ZZP'ers zijn niet ingeschreven in SVH
        is_geregistreerd = False

    registratienummer = f'SVH-{random.randint(100000, 999999)}' if is_geregistreerd else None
    svh_data = {
        'registraties': [{
            'bsn': bsn,
            'is_geregistreerd': is_geregistreerd,
            'registratienummer': registratienummer,
        }],
    }
    if is_horeca:
        svh_data['register_sociale_hygiene'] = [{
            'bsn': bsn,
            'is_geregistreerd': is_geregistreerd,
            'registratienummer': registratienummer,
            'diploma_type': random.choice(['Sociaal Hygienisch Werken', 'Leidinggevende Horeca']) if is_geregistreerd else None,
        }]
    return svh_data


def generate_lbb(bsn, profile_data):
    """Generate LBB (Landelijk Bureau Bibob) Bibob-advies voor horeca (Wet Bibob art. 3).
    Verdeling: geen_gevaar 90%, mindere_mate 8%, ernstig_gevaar 2%.
    """
    if not profile_data.get('is_horeca', False):
        return None

    kans = random.random()
    if kans < 0.90:
        mate = 'geen_gevaar'
    elif kans < 0.98:
        mate = 'mindere_mate'
    else:
        mate = 'ernstig_gevaar'

    return {
        'bibob_adviezen': [{
            'kvk_nummer': profile_data.get('kvk_nummer'),
            'beschikking_type': 'vergunning',
            'mate_van_gevaar': mate,
            'advies_datum': f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
            'advies_uitgebracht': True,
        }]
    }


def generate_uwv(bsn, profile_data):
    """Generate UWV data based on work status."""
    is_employee = profile_data.get('is_employee', False)
    werkloosheid = profile_data.get('werkloosheid', False)

    if is_employee:
        dienstverband = 'VAST' if random.random() > 0.3 else 'TIJDELIJK'
        # CBS 2023: ~52% fulltime (≥36u/week), ~48% parttime (<36u/week)
        is_fulltime = random.random() < 0.52
        uren_week = random.choice([36, 38, 40]) if is_fulltime else random.choice([16, 20, 24, 28, 32])
        jaarloon = profile_data.get('yearly_income', 3500000)
    else:
        dienstverband = 'GEEN'
        uren_week = 0
        jaarloon = 0

    return {
        'arbeidsverhoudingen': [{
            'bsn': bsn,
            'dienstverband_type': dienstverband,
            'verzekerd_ww': is_employee,
            'verzekerd_wia': is_employee
        }],
        'uwv_toetsingsinkomen': [{
            'bsn': bsn,
            'toetsingsinkomen': profile_data.get('yearly_income', 0)
        }],
        'uwv_werkgegevens': [{
            'bsn': bsn,
            'gemiddeld_uren_per_week': float(uren_week),
            'huidige_uren_per_week': float(uren_week),
            'gewerkte_weken_36': 52 if is_employee else 0,
            'arbeidsverleden_jaren': random.randint(0, 30) if is_employee else 0,
            'jaarloon': jaarloon
        }],
        'ziektewet': [{'bsn': bsn, 'heeft_ziektewet_uitkering': False}],
        'WIA': [{'bsn': bsn, 'heeft_wia_uitkering': False}]
    }


def generate_rvz(bsn):
    """Generate RVZ (zorgverzekering) data."""
    return {
        'verzekeringen': [{
            'bsn': bsn,
            'polis_status': 'ACTIEF',
            'verdrag_status': 'GEEN',
            'zorg_type': 'BASIS'
        }]
    }


def generate_rechtspraak(bsn):
    """Generate RECHTSPRAAK data (curatele_registraties).

    Curatele = rechterlijke maatregel waarbij iemand handelingsonbekwaam wordt verklaard.
    In de praktijk ~0.2% van de volwassen bevolking. Voor de meeste profielen lege lijst.
    ~2% kans op curatele voor testdekking van IS_ONDER_CURATELE_EXPLOITANT.
    """
    is_onder_curatele = random.random() < 0.02
    if is_onder_curatele:
        datum_ingang = f"{random.randint(2015, 2023)}-{random.randint(1,12):02d}-01"
        # Curatele is meestal onbepaald (datum_einde None) of beëindigd
        datum_einde = None if random.random() < 0.7 else f"{random.randint(2023, 2025)}-{random.randint(1,12):02d}-01"
        registraties = [{
            'bsn_curandus': bsn,
            'datum_ingang': datum_ingang,
            'datum_einde': datum_einde,
            'status': 'ACTIEF' if datum_einde is None else 'BEEINDIGD',
            'bsn_curator': None,
            'naam_curandus': None,
        }]
    else:
        registraties = []

    return {
        'curatele_registraties': registraties
    }


def generate_duo(bsn, age):
    """Generate DUO data."""
    is_student = 18 <= age <= 30 and random.random() < 0.15  # 15% chance if young

    return {
        'inschrijvingen': [{
            'bsn': bsn,
            'onderwijssoort': 'HBO' if is_student else 'GEEN'
        }],
        'studiefinanciering': [{
            'bsn': bsn,
            'ontvangt_studiefinanciering': is_student
        }],
        'is_student': [{'bsn': bsn, 'waarde': is_student}],
        'receives_study_grant': [{'bsn': bsn, 'waarde': is_student}]
    }


def generate_dji(bsn):
    """Generate DJI (detentie) data - normally empty."""
    return {
        'detenties': [{'bsn': bsn, 'is_gedetineerd': False}],
        'is_detainee': [{'bsn': bsn, 'waarde': False}],
        'detentie': [{'bsn': bsn, 'is_gedetineerd': False}]
    }


def generate_gemeente(bsn, gemeente, profile_data):
    """Generate gemeente data matching the bijstandswet YAML source spec.

    werk_en_re_integratie fields required by YAML:
      arbeidsvermogen, ontheffing_reden, ontheffing_einddatum, re_integratie_traject

    bijstandswet arbeidsverplichting check (OR):
      - arbeidsvermogen IN [MEDISCH_VOLLEDIG, MANTELZORG_VOLLEDIG, SOCIALE_OMSTANDIGHEDEN_VOLLEDIG]
      - re_integratie_traject NOT NULL
    """
    gemeente_key = f"GEMEENTE_{gemeente.upper().replace(' ', '_')}"

    ontheffing_reden = None
    ontheffing_einddatum = None

    # ~8% of werklozen/werknemers hebben een volledige ontheffing wegens medische
    # of zorgredenen; deze voldoen via de eerste OR-tak van de bijstandswet eis.
    heeft_volledige_ontheffing = random.random() < 0.08

    if profile_data.get('is_gepensioneerd'):
        # Gepensioneerden vallen buiten bijstand — geen werk_en_re_integratie rij nodig
        return {gemeente_key: {}}
    elif heeft_volledige_ontheffing:
        # Volledige ontheffing wegens medische/zorg/sociale reden
        arbeidsvermogen = random.choice([
            'MEDISCH_VOLLEDIG', 'MANTELZORG_VOLLEDIG', 'SOCIALE_OMSTANDIGHEDEN_VOLLEDIG'
        ])
        ontheffing_reden = arbeidsvermogen
        if arbeidsvermogen == 'MEDISCH_VOLLEDIG':
            ontheffing_einddatum = (
                f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                if random.random() < 0.6 else None
            )
    else:
        # Geen ontheffing — arbeidsvermogen is niet van toepassing als ontheffingsgrond.
        # Kwalificatie loopt via re_integratie_traject (tweede OR-tak bijstandswet).
        arbeidsvermogen = None

    # Re-integratie traject
    re_integratie = None
    if profile_data.get('werkloosheid') and not heeft_volledige_ontheffing:
        # Werklozen in bijstand hebben vrijwel altijd een re-integratietraject
        # (Participatiewet art. 9 — arbeidsplicht)
        re_integratie = random.choice([
            'Sollicitatietraining', 'Omscholing', 'Werkstage',
            'Sollicitatietraining', 'Omscholing',  # hogere kans op traject
        ])
    elif profile_data.get('is_zzper'):
        re_integratie = 'Ondernemerscoaching' if random.random() < 0.2 else None

    return {
        gemeente_key: {
            'werk_en_re_integratie': [{
                'bsn': bsn,
                'arbeidsvermogen': arbeidsvermogen,
                'ontheffing_reden': ontheffing_reden,
                'ontheffing_einddatum': ontheffing_einddatum,
                're_integratie_traject': re_integratie,
            }]
        }
    }


def generate_svb(bsn, age, profile_data):
    """Generate SVB data."""
    pensioenleeftijd = 68  # Current AOW age

    # Kinderbijslag
    children_data = profile_data.get('children_data', [])
    num_children = len(children_data)
    children_ages = [c.get('age', 0) for c in children_data]
    # Only children under 18 qualify
    eligible_children = [a for a in children_ages if a < 18]

    svb_data = {
        'retirement_age': [{'bsn': bsn, 'leeftijd': pensioenleeftijd}],
        'algemene_ouderdomswet_gegevens': [{
            'bsn': bsn,
            'pensioenleeftijd': pensioenleeftijd
        }]
    }

    if eligible_children:
        svb_data['algemene_kinderbijslagwet'] = [{
            'ouder_bsn': bsn,
            'aantal_kinderen': len(eligible_children),
            'kinderen_leeftijden': eligible_children,
            'ontvangt_kinderbijslag': True
        }]

    return svb_data


def generate_ind(bsn):
    """Generate IND data - for Dutch nationals."""
    return {
        'verblijfsvergunningen': [{
            'bsn': bsn,
            'type': 'PERMANENT',
            'status': 'VERLEEND'
        }],
        'residence_permit_type': [{
            'bsn': bsn,
            'type': 'PERMANENT'
        }],
        'vreemdelingenwet': [{
            'bsn': bsn,
            'verblijfsvergunning_type': None
        }]
    }


def generate_szw(bsn, profile_data):
    """Generate SZW data.

    - ZZP'ers: bbz_aanvraag tabel
    - Werklozen / lage-inkomen profielen: bijstand tabel met is_gerechtigd,
      basisbedrag en kostendelersnorm (vereist door gemeente-bijstand YAML als
      service_reference naar SZW/participatiewet/bijstand)
    """
    szw = {}

    # Bijstand tabel — voor werklozen en bijstand-doelgroep
    if profile_data.get('werkloosheid') or profile_data.get('bijstand_doelgroep'):
        heeft_partner = profile_data.get('has_partner', False)
        household_size = profile_data.get('household_size', 1)

        # Basisbedrag (eurocent/mnd): alleenstaand 1089 euro, partners 1556 euro
        basisbedrag = 155600 if heeft_partner else 108900

        # Kostendelersnorm op basis van huishoudgrootte (Participatiewet art. 22a)
        kostendelers_factoren = {1: 1.00, 2: 0.50, 3: 0.43, 4: 0.40}
        kostendelersnorm = kostendelers_factoren.get(household_size, 0.38)

        szw['bijstand'] = [{
            'bsn': bsn,
            'is_gerechtigd': True,
            'basisbedrag': basisbedrag,
            'kostendelersnorm': kostendelersnorm,
        }]

    # Bbz tabel — alleen voor ZZP'ers
    if profile_data.get('is_zzper', False):
        szw['bbz_aanvraag'] = [{
            'bsn': bsn,
            'type_zelfstandige': 'GEVESTIGD',
            'bedrijf_levensvatbaar': True,
            'jaren_ondernemerschap': random.randint(1, 15),
            'uren_per_week': random.choice([24, 32, 40]),
            'beeindigingsdatum': None
        }]

    return szw if szw else None


def generate_profile(bsn, global_services, used_bsns, template_profile=None,
                     force_male=None, force_age_group=None, force_background=None,
                     force_work_status=None, force_income_level=None,
                     force_has_partner=None, force_has_children=None):
    """Generate a single profile with optional forced characteristics for balanced distribution."""

    # Determine gender (can be forced for 50/50 balance)
    if force_male is not None:
        is_male = force_male
    else:
        is_male = random.choice([True, False])

    # Determine background (can be forced for diversity)
    if force_background is not None:
        background = force_background
    else:
        background = random.choice(BACKGROUNDS)

    # Get name matching background
    first_name, last_name = get_name_for_background(background, is_male)
    full_name = f"{first_name} {last_name}"

    # Generate age and birth date (can use specific age group)
    if force_age_group is not None:
        geboortedatum, age = generate_geboortedatum(age_group=force_age_group)
    else:
        geboortedatum, age = generate_geboortedatum()

    # Choose gemeente
    gemeente = random.choice(list(GEMEENTEN.keys()))
    address_data = generate_address(gemeente)

    # Partner status (can be forced for 50/50 balance)
    if force_has_partner is not None:
        has_partner = force_has_partner
    else:
        has_partner = random.choice([True, False])

    # Children status (can be forced for balanced distribution)
    if force_has_children is not None:
        has_children = force_has_children
    else:
        has_children = random.choice([True, False, False])  # 33% chance
    num_children = random.randint(1, 3) if has_children else 0

    # Work status (can be forced for 25% distribution each)
    if force_work_status is not None:
        status_choice = force_work_status
    else:
        status_choice = random.choice(WORK_STATUSES)

    is_zzper = status_choice == 'zzp'
    is_employee = status_choice == 'employee'
    werkloosheid = status_choice == 'werkloosheid'
    is_gepensioneerd = status_choice == 'gepensioneerd' or age > 65

    # Income level (can be forced for ~33% distribution each)
    if force_income_level is not None:
        income_level = force_income_level
    else:
        income_level = random.choice(INCOME_LEVELS)

    laag_inkomen = income_level == 'laag'
    hoog_inkomen = income_level == 'hoog'

    # Bereken inkomen op basis van CBS-gemiddelden per leeftijdsklasse.
    # Werklozen zonder WW (bijstand-doelgroep) krijgen 0 inkomen en laag vermogen
    # (~50% kans: WW uitgeput of nooit gekwalificeerd).
    bijstand_doelgroep = werkloosheid and random.random() < 0.50
    if bijstand_doelgroep:
        yearly_income = 0
        laag_inkomen = True  # zorgt ook voor laag vermogen in generate_belastingdienst
    else:
        yearly_income = get_cbs_income(age, income_level)

    # Profile data for description and other generators
    profile_data = {
        'is_zzper': is_zzper,
        'is_employee': is_employee,
        'werkloosheid': werkloosheid,
        'bijstand_doelgroep': bijstand_doelgroep,
        'is_gepensioneerd': is_gepensioneerd,
        'laag_inkomen': laag_inkomen,
        'hoog_inkomen': hoog_inkomen,
        'has_partner': has_partner,
        'has_children': has_children,
        'num_children': num_children,
        'child_with_condition': has_children and random.choice([True, False]),
        'yearly_income': yearly_income,
        'gemeente': gemeente,
        'children_data': []  # Will be populated below
    }

    description = generate_description(profile_data)

    # Household size
    household_size = 1
    if has_partner:
        household_size += 1
    household_size += num_children
    profile_data['household_size'] = household_size

    # Build profile structure with all sources
    profile = {
        'name': full_name,
        'description': description,
        'sources': {
            'CBS': global_services['CBS'],
            'KIESRAAD': global_services['KIESRAAD'],
            'JenV': global_services['JenV'],
            'RvIG': {
                'personen': [{
                    'bsn': bsn,
                    'geboortedatum': geboortedatum,
                    'verblijfsadres': gemeente,
                    'land_verblijf': 'NEDERLAND',
                    'nationaliteit': 'NEDERLANDS',
                    'leeftijd': age,
                    'age': age,  # backwards-compat
                    'geslacht': 'M' if is_male else 'V',
                    'herkomst': {
                        'NL': 'NL_AUTOCHTOON', 'AR': 'MAROKKAANS', 'TR': 'TURKS',
                        'SR': 'SURINAAMS', 'AS': 'AZIATISCH', 'EE': 'OOST_EUROPEES',
                    }.get(background, 'ONBEKEND'),
                    'has_dutch_nationality': True,
                    'has_partner': has_partner,
                    'residence_address': address_data['full_address'],
                    'has_fixed_address': True,
                    'household_size': household_size
                }],
                'relaties': [{
                    'bsn': bsn,
                    'partnerschap_type': 'GEHUWD' if has_partner else 'GEEN',
                    'partner_bsn': generate_bsn(used_bsns) if has_partner else None,
                    'has_partner': has_partner,
                    'kinderen': []
                }],
                'verblijfplaats': [{
                    'bsn': bsn,
                    'straat': address_data['straat'],
                    'huisnummer': address_data['huisnummer'],
                    'postcode': address_data['postcode'],
                    'woonplaats': address_data['woonplaats'],
                    'type': 'WOONADRES',
                    'heeft_vast_adres': True,
                }],
                'brp_gegevens': [{
                    'bsn': bsn,
                    'leeftijd': age,
                    'geboortedatum': geboortedatum,
                    'heeft_vast_adres': True,
                    'heeft_nederlandse_nationaliteit': True,
                    'heeft_partner': has_partner,
                    'verblijfsadres': address_data['full_address'],
                    'huishoudgrootte': household_size,
                }]
            }
        }
    }

    # Add children if applicable
    children_for_svb = []
    if has_children:
        children_data = []
        gezag_relaties = []

        for i in range(num_children):
            child_bsn = generate_bsn(used_bsns)
            child_age = random.randint(0, 17)
            child_birthdate = f"{datetime.now().year - child_age}-{random.randint(1,12):02d}-01"
            child_name = f"Kind {i+1} {last_name}"

            child_info = {
                'geboortedatum': child_birthdate,
                'kind_bsn': child_bsn,
                'age': child_age
            }

            # Random condition for first child sometimes
            if i == 0 and profile_data['child_with_condition']:
                child_info['zorgbehoefte'] = True

            children_data.append(child_info)
            children_for_svb.append({'age': child_age})

            profile['sources']['RvIG']['relaties'][0]['kinderen'].append({
                'bsn': child_bsn,
                'geboortedatum': child_birthdate,
                'naam': child_name
            })

            # Add gezag relatie
            gezag_relaties.append({
                'bsn_gezagdrager': bsn,
                'bsn_kind': child_bsn,
                'naam_kind': child_name,
                'geboortedatum_kind': child_birthdate,
                'type_gezag': 'OUDERLIJK_GEZAG',
                'datum_ingang': child_birthdate,
                'datum_einde': None,
                'status': 'ACTIEF'
            })

        profile['sources']['RvIG']['CHILDREN_DATA'] = [{
            'bsn': bsn,
            'kinderen': children_data
        }]

        profile['sources']['RvIG']['gezag_relaties'] = gezag_relaties
        profile_data['children_data'] = children_for_svb

    # Add BELASTINGDIENST data
    profile['sources']['BELASTINGDIENST'] = generate_belastingdienst(bsn, profile_data)

    # Add KVK data (only for ZZP'ers)
    kvk_data = generate_kvk(bsn, profile_data, full_name)
    if kvk_data:
        profile['sources']['KVK'] = kvk_data

    # Add SVH data (alleen voor horeca ZZP'ers — Alcoholwet art. 8 lid 4)
    svh_data = generate_svh(bsn, profile_data)
    if svh_data:
        profile['sources']['SVH'] = svh_data

    # Add LBB/Bibob data (alleen voor horeca ZZP'ers — Wet Bibob art. 3)
    lbb_data = generate_lbb(bsn, profile_data)
    if lbb_data:
        profile['sources']['LBB'] = lbb_data

    # Add UWV data
    profile['sources']['UWV'] = generate_uwv(bsn, profile_data)

    # Add RVZ data
    profile['sources']['RVZ'] = generate_rvz(bsn)

    # Add DUO data
    profile['sources']['DUO'] = generate_duo(bsn, age)

    # Add DJI data
    profile['sources']['DJI'] = generate_dji(bsn)

    # Add GEMEENTE data
    gemeente_data = generate_gemeente(bsn, gemeente, profile_data)
    profile['sources'].update(gemeente_data)

    # Add SVB data
    profile['sources']['SVB'] = generate_svb(bsn, age, profile_data)

    # Add IND data
    profile['sources']['IND'] = generate_ind(bsn)

    # Add RECHTSPRAAK data (curatele_registraties — vereist voor IS_ONDER_CURATELE_EXPLOITANT)
    profile['sources']['RECHTSPRAAK'] = generate_rechtspraak(bsn)

    # Add SZW data (only for ZZP'ers)
    szw_data = generate_szw(bsn, profile_data)
    if szw_data:
        profile['sources']['SZW'] = szw_data

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Generate profiles based on existing profiles.yaml patterns (NO LLM)"
    )
    parser.add_argument("--input", default="data/profielen/profiles.yaml", help="Input profiles.yaml")
    parser.add_argument("--count", type=int, default=10, help="Number of profiles to generate")
    parser.add_argument("--output", help="Output YAML file (optional, auto-generates in data/profielen/)")
    parser.add_argument("--start-bsn", type=int, default=100000100, help="Starting BSN number")

    args = parser.parse_args()

    # Auto-generate output filename with metadata if not specified
    if not args.output:
        # Create data/profielen directory if it doesn't exist
        output_dir = Path("data/profielen")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with metadata: profiles_COUNT_YYYYMMDD_HHMMSS.yaml
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profiles_{args.count}_{timestamp}.yaml"
        args.output = str(output_dir / filename)

    # Load existing profiles to get global services and structure
    print(f"Loading existing profiles from {args.input}...")
    global_services, existing_profiles = load_existing_profiles(args.input)

    print(f"Found {len(existing_profiles)} existing profiles")
    print(f"Generating {args.count} new profiles...")

    # Track used BSNs
    used_bsns = set(existing_profiles.keys())

    # Generate new profiles with CBS-weighted distributions
    new_profiles = {}

    num_profiles = args.count

    # CBS-gewogen werkstatus (employee ~60%, gepensioneerd ~27%, zzp ~9%, werkloos ~3%)
    work_statuses_list = random.choices(_WS_KEYS, weights=_WS_PROBS, k=num_profiles)

    # Gebalanceerde inkomensniveaus (~33% elk)
    income_levels_list = []
    profiles_per_income = num_profiles // len(INCOME_LEVELS)
    remainder_income = num_profiles % len(INCOME_LEVELS)
    for i, level in enumerate(INCOME_LEVELS):
        count = profiles_per_income + (1 if i < remainder_income else 0)
        income_levels_list.extend([level] * count)
    random.shuffle(income_levels_list)

    # CBS-gewogen herkomst (75.1% NL-autochtoon + 6.4% één buitenlandse ouder + 18.5% migrantenachtergrond)
    backgrounds_list = random.choices(_BG_KEYS, weights=_BG_PROBS, k=num_profiles)

    # CBS-gewogen leeftijd en geslacht — ZZP'ers krijgen eigen CBS-verdeling
    num_zzp = work_statuses_list.count('zzp')
    num_non_zzp = num_profiles - num_zzp

    # ZZP: 45-75j 60%, 25-45j 35.1%, 15-25j 4.9%  |  62% man, 38% vrouw
    zzp_ages    = random.choices(ZZP_AGE_GROUPS, weights=ZZP_AGE_WEIGHTS, k=num_zzp)
    zzp_genders = random.choices([True, False], weights=ZZP_GENDER_WEIGHTS, k=num_zzp)

    # Koppel leeftijd/geslacht aan werkstatus — per status eigen verdeling
    # zodat gepensioneerden altijd 65+ krijgen en werkenden altijd 18-64.
    num_gepens   = work_statuses_list.count('gepensioneerd')
    num_werkend  = num_non_zzp - num_gepens  # loondienst + werkloosheid

    gepens_ages  = random.choices(GEPENSIONEERD_AGE_GROUPS,  weights=GEPENSIONEERD_AGE_WEIGHTS,  k=num_gepens)
    werkend_ages = random.choices(WERKEND_AGE_GROUPS,        weights=WERKEND_AGE_WEIGHTS,         k=num_werkend)
    non_zzp_genders = [random.choice([True, False]) for _ in range(num_non_zzp)]

    age_groups_list = []
    genders_list = []
    zzp_i = gepens_i = werkend_i = non_zzp_gender_i = 0
    for status in work_statuses_list:
        if status == 'zzp':
            age_groups_list.append(zzp_ages[zzp_i])
            genders_list.append(zzp_genders[zzp_i])
            zzp_i += 1
        elif status == 'gepensioneerd':
            age_groups_list.append(gepens_ages[gepens_i])
            genders_list.append(non_zzp_genders[non_zzp_gender_i])
            gepens_i += 1
            non_zzp_gender_i += 1
        else:
            age_groups_list.append(werkend_ages[werkend_i])
            genders_list.append(non_zzp_genders[non_zzp_gender_i])
            werkend_i += 1
            non_zzp_gender_i += 1

    # 50/50 partnerstatus
    num_with_partner = num_profiles // 2
    num_without_partner = num_profiles - num_with_partner
    partner_list = [True] * num_with_partner + [False] * num_without_partner
    random.shuffle(partner_list)

    # 33/67 kinderenstatus (33% met kinderen)
    num_with_children = num_profiles // 3
    num_without_children = num_profiles - num_with_children
    children_list = [True] * num_with_children + [False] * num_without_children
    random.shuffle(children_list)

    num_male = sum(genders_list)
    num_female = num_profiles - num_male
    bg_counts = {k: backgrounds_list.count(k) for k in _BG_KEYS if backgrounds_list.count(k) > 0}
    print(f"\n  CBS-gewogen verdeling:")
    print(f"    Geslacht: {num_male} man, {num_female} vrouw")
    print(f"    Leeftijdsgroepen: CBS-gewogen (ZZP: {ZZP_AGE_WEIGHTS}%, overig: {AGE_GROUP_WEIGHTS}%)")
    print(f"    Herkomst: {', '.join(f'{k}:{v}' for k, v in bg_counts.items())}")
    ws_counts = {k: work_statuses_list.count(k) for k in _WS_KEYS if work_statuses_list.count(k) > 0}
    print(f"    Werkstatus (CBS): {', '.join(f'{k}:{v}' for k, v in ws_counts.items())}")
    print(f"    Inkomen: {', '.join(INCOME_LEVELS)} (~{profiles_per_income} elk, CBS-gewogen per leeftijd)")
    print(f"    Partner: {num_with_partner} ja, {num_without_partner} nee")
    print(f"    Kinderen: {num_with_children} ja, {num_without_children} nee")

    for i in range(num_profiles):
        bsn = generate_bsn(used_bsns)
        profile = generate_profile(
            bsn, global_services, used_bsns,
            force_male=genders_list[i],
            force_age_group=age_groups_list[i],
            force_background=backgrounds_list[i],
            force_work_status=work_statuses_list[i],
            force_income_level=income_levels_list[i],
            force_has_partner=partner_list[i],
            force_has_children=children_list[i]
        )
        new_profiles[bsn] = profile

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.count} profiles...")

    # Create output structure
    output_data = {
        'globalServices': global_services,
        'profiles': new_profiles
    }

    # Write to file
    output_path = Path(args.output)
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nGenerated {len(new_profiles)} profiles")
    print(f"Saved to: {output_path.absolute()}")

    # Show sample
    print("\n" + "="*80)
    print("SAMPLE PROFILES")
    print("="*80)
    for i, (bsn, profile) in enumerate(list(new_profiles.items())[:3], 1):
        person = profile['sources']['RvIG']['personen'][0]
        print(f"\n{i}. {profile['name']} (BSN: {bsn})")
        print(f"   {profile['description']}")
        print(f"   Leeftijd: {person['age']}, Woonplaats: {person['verblijfsadres']}")
        print(f"   Huishouden: {person['household_size']} personen")


if __name__ == "__main__":
    main()
