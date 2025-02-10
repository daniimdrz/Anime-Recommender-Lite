import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import sklearn
from fuzzywuzzy import fuzz

# =======================
# Datos de ejemplo (versión lite)
# =======================
# Cada anime debe contar con las siguientes claves:
# 'title', 'main_picture_medium', 'genres', 'num_episodes', 'rating' y 'synopsis'
ANIMES = [
    {
        "title": "Naruto",
        "main_picture_medium": "https://upload.wikimedia.org/wikipedia/en/9/94/NarutoCoverTankobon1.jpg",
        "genres": "Action, Adventure, Martial Arts",
        "num_episodes": 220,
        "rating": 8.2,
        "synopsis": (
            "Naruto Uzumaki, a spirited ninja in training, struggles to overcome his past and earn recognition from his peers "
            "while striving to become the strongest ninja in the Hidden Leaf Village. Filled with action-packed battles, deep personal growth, "
            "and a richly detailed world, Naruto’s journey inspires those around him as he confronts formidable foes and discovers the true meaning "
            "of friendship and sacrifice."
        )
    },
    {
        "title": "One Piece",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/5/19082l.jpg?_gl=1*1bem8jn*_gcl_au*MTcxNTUyNDY3Mi4xNzM5MTg1MzMy*_ga*MTA2Mzk3NjUuMTczOTE4NTMzMA..*_ga_26FEP9527K*MTczOTE4NTMzMS4xLjEuMTczOTE4NTM1NC4zNy4wLjA.",
        "genres": "Adventure, Fantasy, Action",
        "num_episodes": 1000,
        "rating": 8.9,
        "synopsis": (
            "Set sail with Monkey D. Luffy and his diverse crew of pirates on an epic journey across the Grand Line in search of the legendary treasure "
            "known as One Piece. The series masterfully blends humor, adventure, and heartfelt drama while exploring themes of freedom, friendship, "
            "and the relentless pursuit of dreams in a vibrant world filled with mystery and danger."
        )
    },
    {
        "title": "Attack on Titan",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/10/47347l.jpg",
        "genres": "Action, Drama, Mystery",
        "num_episodes": 75,
        "rating": 9.0,
        "synopsis": (
            "In a world where humanity teeters on the brink of extinction behind colossal walls, the sudden appearance of giant humanoid Titans shatters the illusion of safety. "
            "Attack on Titan follows Eren Yeager and his comrades as they join the military to fight the relentless Titans, uncovering dark secrets about their world "
            "and the true nature of the enemy lurking beyond the walls."
        )
    },
    {
        "title": "Death Note",
        "main_picture_medium": "https://upload.wikimedia.org/wikipedia/en/6/6f/Death_Note_Vol_1.jpg",
        "genres": "Mystery, Supernatural, Thriller",
        "num_episodes": 37,
        "rating": 9.0,
        "synopsis": (
            "Death Note chronicles the chilling tale of Light Yagami, a brilliant student who stumbles upon a mysterious notebook that grants him the power to kill anyone "
            "by simply writing their name. As Light embarks on a crusade to create a new world free of crime, he engages in a high-stakes battle of wits with a legendary detective, "
            "questioning the boundaries of justice and morality in a dark, psychological thriller."
        )
    },
    {
        "title": "Fullmetal Alchemist: Brotherhood",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/7/74317l.jpg",
        "genres": "Adventure, Drama, Fantasy",
        "num_episodes": 64,
        "rating": 9.1,
        "synopsis": (
            "In a world where alchemy is a revered science, brothers Edward and Alphonse Elric embark on a perilous journey to restore their bodies after a forbidden alchemical experiment goes horribly wrong. "
            "Fullmetal Alchemist: Brotherhood weaves a complex tale of sacrifice, redemption, and the unyielding quest for truth, as the brothers confront the harsh realities of power, corruption, "
            "and the human condition."
        )
    },
    {
        "title": "Dragon Ball Z",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/6/20936l.jpg?_gl=1*1h9mazm*_gcl_au*MTcxNTUyNDY3Mi4xNzM5MTg1MzMy*_ga*MTA2Mzk3NjUuMTczOTE4NTMzMA..*_ga_26FEP9527K*MTczOTE4NTMzMS4xLjEuMTczOTE4NTY0MS41Ny4wLjA.",
        "genres": "Action, Adventure, Martial Arts",
        "num_episodes": 291,
        "rating": 8.5,
        "synopsis": (
            "Dragon Ball Z follows the adventures of Goku and his friends as they defend Earth against a series of increasingly powerful foes. "
            "From epic battles to transformative power-ups, the series captivates audiences with its dynamic fight scenes, charismatic characters, and the timeless theme of the hero's journey "
            "in a universe where strength and determination are key."
        )
    },
    {
        "title": "Cowboy Bebop",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/7/3791.jpg",
        "genres": "Action, Sci-Fi, Drama",
        "num_episodes": 26,
        "rating": 8.9,
        "synopsis": (
            "Set in a futuristic world where bounty hunters roam the galaxy, Cowboy Bebop follows the enigmatic Spike Spiegel and his ragtag crew aboard the spaceship Bebop. "
            "Blending jazz-infused music with stylish animation and philosophical undertones, the series explores themes of loneliness, existentialism, and the pursuit of redemption "
            "in a universe as vast as it is unpredictable."
        )
    },
    {
        "title": "Steins;Gate",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/1935/127974l.jpg?_gl=1*e1a295*_gcl_au*MTcxNTUyNDY3Mi4xNzM5MTg1MzMy*_ga*MTA2Mzk3NjUuMTczOTE4NTMzMA..*_ga_26FEP9527K*MTczOTE4NTMzMS4xLjEuMTczOTE4NTcwNC42MC4wLjA.",
        "genres": "Sci-Fi, Thriller, Drama",
        "num_episodes": 24,
        "rating": 9.1,
        "synopsis": (
            "A gripping tale of time travel and its consequences, Steins;Gate follows self-proclaimed mad scientist Rintarou Okabe and his friends as they accidentally discover a method of sending messages into the past. "
            "As they unravel the complexities of time, their actions trigger a series of unforeseen events, forcing them to confront the delicate balance between fate and free will in a suspenseful race against time."
        )
    },
    {
        "title": "My Hero Academia",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/10/78745l.jpg?_gl=1*192oab9*_gcl_au*MTcxNTUyNDY3Mi4xNzM5MTg1MzMy*_ga*MTA2Mzk3NjUuMTczOTE4NTMzMA..*_ga_26FEP9527K*MTczOTE4NTMzMS4xLjEuMTczOTE4NTczNS4yOS4wLjA.",
        "genres": "Action, Superhero, Comedy",
        "num_episodes": 113,
        "rating": 8.5,
        "synopsis": (
            "In a world where superpowers, known as Quirks, are the norm, My Hero Academia follows Izuku Midoriya, a Quirkless boy with dreams of becoming a great hero. "
            "Through rigorous training and unwavering determination, Izuku enrolls in an elite academy for aspiring heroes, facing formidable challenges and learning the true meaning of courage, friendship, and sacrifice."
        )
    },
    {
        "title": "Demon Slayer: Kimetsu no Yaiba",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/1286/99889l.jpg",
        "genres": "Action, Dark Fantasy, Historical",
        "num_episodes": 26,
        "rating": 8.7,
        "synopsis": (
            "Demon Slayer follows Tanjiro Kamado, a kind-hearted young boy who becomes a demon slayer after his family is slaughtered by demons and his sister Nezuko is transformed into one. "
            "Set in Taisho-era Japan, the series is a visually stunning and emotionally charged journey of vengeance, hope, and the enduring bond between siblings in the face of overwhelming darkness."
        )
    },
    {
        "title": "Hunter x Hunter (2011)",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/1337/99013l.jpg?_gl=1*1uzx3b*_gcl_au*MTcxNTUyNDY3Mi4xNzM5MTg1MzMy*_ga*MTA2Mzk3NjUuMTczOTE4NTMzMA..*_ga_26FEP9527K*MTczOTE4NTMzMS4xLjEuMTczOTE4NTc5OS42MC4wLjA.",
        "genres": "Adventure, Fantasy, Action",
        "num_episodes": 148,
        "rating": 9.0,
        "synopsis": (
            "Hunter x Hunter follows Gon Freecss, a young boy who embarks on an extraordinary journey to become a Hunter and find his missing father. "
            "Along the way, he encounters a colorful cast of characters and faces perilous challenges, delving into a world where the pursuit of one's dreams is fraught with danger, intrigue, "
            "and unexpected alliances."
        )
    },
    {
        "title": "Code Geass: Lelouch of the Rebellion",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/10/50329l.jpg",
        "genres": "Mecha, Drama, Thriller",
        "num_episodes": 50,
        "rating": 8.8,
        "synopsis": (
            "In a dystopian future ruled by oppressive forces, Code Geass follows Lelouch vi Britannia, a brilliant strategist who acquires the power of Geass—a mysterious ability that compels absolute obedience. "
            "With his newfound power, Lelouch leads a rebellion against a tyrannical empire, orchestrating a complex battle of wits, morality, and sacrifice that challenges the very fabric of society."
        )
    },
    {
        "title": "Sword Art Online",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/11/61515l.jpg",
        "genres": "Action, Adventure, Sci-Fi",
        "num_episodes": 25,
        "rating": 7.6,
        "synopsis": (
            "Sword Art Online immerses viewers in a virtual reality MMORPG where players become trapped in the game. "
            "Kirito, a skilled gamer, must navigate deadly challenges and forge unexpected alliances to escape the digital world, blurring the lines between virtual and reality in a high-stakes battle for survival."
        )
    },
    {
        "title": "Bleach",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/3/24168l.jpg",
        "genres": "Action, Adventure, Supernatural",
        "num_episodes": 366,
        "rating": 8.0,
        "synopsis": (
            "Bleach follows Ichigo Kurosaki, a teenager with the ability to see ghosts, as he inadvertently obtains the powers of a Soul Reaper. "
            "Thrust into a world of supernatural battles and spiritual intrigue, Ichigo and his friends confront powerful adversaries and unravel the mysteries of the afterlife in a series that masterfully blends action, emotion, and supernatural elements."
        )
    },
    {
        "title": "Neon Genesis Evangelion",
        "main_picture_medium": "https://cdn.myanimelist.net/images/anime/12/21418l.jpg?_gl=1*1uref0g*_gcl_au*MTcxNTUyNDY3Mi4xNzM5MTg1MzMy*_ga*MTA2Mzk3NjUuMTczOTE4NTMzMA..*_ga_26FEP9527K*MTczOTE4NTMzMS4xLjEuMTczOTE4NTkyNS4zLjAuMA..",
        "genres": "Mecha, Psychological, Drama",
        "num_episodes": 26,
        "rating": 8.5,
        "synopsis": (
            "Neon Genesis Evangelion delves into the psychological and existential struggles of a group of teenagers piloting giant mechas known as Evangelions to protect Earth from mysterious beings called Angels. "
            "This groundbreaking series challenges conventional storytelling with its complex characters, deep symbolism, and exploration of themes such as identity, isolation, and the burden of responsibility."
        )
    }
]

# =======================
# Funciones de procesamiento y recomendación
# =======================

def categorizar_duracion(num_episodios):
    """
    Categoriza la duración de un anime según el número de episodios.
    """
    if num_episodios <= 13:
        return 'Corta'
    elif num_episodios <= 26:
        return 'Media'
    else:
        return 'Larga'

def prepare_data(df):
    """
    Prepara el DataFrame añadiendo la columna 'Duración',
    codifica los géneros y la duración, y genera una matriz de similitud.
    """
    # Agregar columna de duración
    df['Duración'] = df['num_episodes'].apply(categorizar_duracion)
    
    # Verificar la versión de scikit-learn para usar el parámetro correcto
    if sklearn.__version__ >= '0.24':
        encoder = OneHotEncoder(sparse_output=False)
    else:
        encoder = OneHotEncoder(sparse=False)
    
    # Codificar los géneros y la duración (se espera que sean strings)
    genero_encoded = encoder.fit_transform(df[['genres']])
    duracion_encoded = encoder.fit_transform(df[['Duración']])
    
    # Convertir la calificación (rating) a float32 para optimizar memoria
    rating_array = df[['rating']].values.astype(np.float32)
    
    # Concatenar las características: rating, géneros y duración
    matriz_caracteristicas = np.hstack((
        rating_array,
        genero_encoded.astype(np.float32),
        duracion_encoded.astype(np.float32)
    ))
    
    # Calcular la similitud coseno entre los vectores de características
    matriz_similitud = cosine_similarity(matriz_caracteristicas)
    similarity_df = pd.DataFrame(matriz_similitud, index=df['title'], columns=df['title'])
    
    return df, similarity_df

def es_similar_nombre(nombre1, nombre2, umbral=90):
    """
    Determina si dos nombres son similares usando fuzzy matching.
    """
    return fuzz.ratio(nombre1.lower(), nombre2.lower()) > umbral

# Caché simple para almacenar recomendaciones y evitar cálculos repetidos.
_recommendations_cache = {}

def recomendar_animes(anime, df, similarity_df, umbral_similitud=90):
    """
    Retorna una lista de recomendaciones para el anime solicitado.
    Utiliza un caché para no recalcular recomendaciones ya solicitadas.
    """
    cache_key = anime.lower()
    if cache_key in _recommendations_cache:
        return _recommendations_cache[cache_key]
    
    anime_lower = anime.lower()
    # Buscar el título en el DataFrame (comparación en minúsculas)
    titles_lower = similarity_df.index.str.lower()
    if anime_lower not in titles_lower:
        result = [{"title": "El anime no está en la base de datos."}]
        _recommendations_cache[cache_key] = result
        return result
    
    # Recuperar el título exacto según la coincidencia
    anime_exacto = similarity_df.index[titles_lower == anime_lower].tolist()[0]
    # Obtener los 6 animes más similares (excluyendo el mismo)
    similar_animes = similarity_df[anime_exacto].sort_values(ascending=False).iloc[1:7]
    
    recomendados = []
    for nombre in similar_animes.index:
        if not any(es_similar_nombre(nombre, recomendado, umbral_similitud) for recomendado in recomendados):
            recomendados.append(nombre)
    
    if not recomendados:
        result = [{"title": "No hay recomendaciones disponibles debido a la similitud de nombres."}]
        _recommendations_cache[cache_key] = result
        return result
    
    recomendaciones = []
    for nombre in recomendados[:6]:
        # Se obtiene la información del anime para la recomendación
        anime_info = df[df['title'] == nombre].iloc[0]
        recomendaciones.append({
            "title": nombre,
            "main_picture_medium": anime_info["main_picture_medium"],
            "rating": float(anime_info["rating"]),
            "num_episodes": int(anime_info["num_episodes"]),
            "synopsis": anime_info["synopsis"]
        })
    
    _recommendations_cache[cache_key] = recomendaciones
    return recomendaciones

# =======================
# Configuración de la aplicación Flask
# =======================

app = Flask(__name__,
            template_folder=os.path.join('..', 'frontend'),
            static_folder=os.path.join('..', 'frontend', 'static'))

# Convertir el diccionario de animes a un DataFrame y preparar los datos.
print("Cargando datos (versión lite, datos manuales)...")
df = pd.DataFrame(ANIMES)
df, similarity_df = prepare_data(df)
print("Datos cargados exitosamente.")

# Ruta principal: renderiza la página de inicio.
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para autocompletar títulos de anime.
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    
    query_lower = query.lower()
    # Filtrar títulos que contengan la consulta (sin distinguir mayúsculas)
    matches = df[df['title'].str.lower().str.contains(query_lower)]
    suggestions = matches[['title', 'main_picture_medium']].drop_duplicates().head(4)
    suggestions_list = suggestions.to_dict(orient='records')
    return jsonify(suggestions_list)

# Ruta para obtener recomendaciones basadas en el título del anime.
@app.route('/recomendaciones', methods=['GET'])
def obtener_recomendaciones():
    anime = request.args.get('anime', '')
    if anime:
        recomendaciones = recomendar_animes(anime, df, similarity_df)
        return jsonify(recomendaciones)
    else:
        return jsonify({"message": "Por favor, proporcione un nombre de anime para obtener recomendaciones."})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
