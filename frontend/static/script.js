// Reiniciar el formulario y ocultar contenedores al cargar la página
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById("anime-form").reset();
    document.getElementById("anime-list").innerHTML = "";
    document.getElementById("recommendations-container").style.display = 'none';
    document.getElementById("selected-anime-container").style.display = 'none';
    document.getElementById("autocomplete-container").style.display = 'none';
});

// ----------------------------
// Función para obtener y mostrar recomendaciones
// ----------------------------
function getRecommendations(animeTitle) {
    if (!animeTitle) return; 
    
    // Ocultar autocomplete si está visible
    document.getElementById('autocomplete-container').style.display = 'none';

    // Ocultar el botón de "Get recommendations"
    document.querySelector('button[type="submit"]').style.display = 'none';

    // Petición al endpoint de recomendaciones
    fetch(`/recomendaciones?anime=${encodeURIComponent(animeTitle)}`)
        .then(response => response.json())
        .then(data => {
            allRecommendations = data;
            const animeList = document.getElementById('anime-list');
            const recommendationsContainer = document.getElementById('recommendations-container');

            // Limpiar contenido previo
            animeList.innerHTML = '';

            if (allRecommendations && allRecommendations.length > 0) {
                recommendationsContainer.style.display = 'block';
                // Renderizar las primeras 3 recomendaciones
                renderRecommendations(0, 3);
                // Crear el botón "Mostrar más" si existen más recomendaciones
                createShowMoreButton(3);
            } else {
                recommendationsContainer.style.display = 'none';
                animeList.innerHTML = '<div>No se encontraron recomendaciones.</div>';
            }
        })
        .catch(error => {
            console.error('Error al obtener las recomendaciones:', error);
        });

    // Limpiar el input de búsqueda
    document.getElementById('anime-title').value = '';
}

// ----------------------------
// AUTOCOMPLETADO
// ----------------------------
document.getElementById('anime-title').addEventListener('input', function(event) {
    const query = event.target.value;
    const container = document.getElementById('autocomplete-container');

    // Si no hay texto, limpiar y ocultar el contenedor de sugerencias
    if(query.length === 0) {
        container.innerHTML = '';
        container.style.display = 'none';
        return;
    }

    // Petición al backend para obtener sugerencias
    fetch(`/autocomplete?query=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(suggestions => {
            container.innerHTML = '';
            if (suggestions.length > 0) {
                container.style.display = 'flex';
                suggestions.forEach(suggestion => {
                    const item = document.createElement('div');
                    item.classList.add('autocomplete-item');
                    item.innerHTML = `
                        <img src="${suggestion.main_picture_medium}" alt="${suggestion.title}" width="100">
                        <div class="autocomplete-title">${suggestion.title}</div>
                    `;
                    // Al hacer clic, se completa el input y se solicitan las recomendaciones directamente
                    item.addEventListener('click', function() {
                        // Actualizamos el input (opcional)
                        document.getElementById('anime-title').value = suggestion.title;
                        // Limpiar y ocultar el contenedor de autocompletado
                        container.innerHTML = '';
                        container.style.display = 'none';
                        // Solicitar las recomendaciones inmediatamente
                        getRecommendations(suggestion.title);
                    });
                    container.appendChild(item);
                });
            } else {
                container.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error fetching autocomplete suggestions:', error);
            container.style.display = 'none';
        });
});

// ----------------------------
// BORRAR RECOMENDACIONES AL ESCRIBIR
// ----------------------------
document.getElementById('anime-title').addEventListener('input', function() {
    // Limpiar las recomendaciones si el usuario empieza a escribir algo nuevo
    document.getElementById('anime-list').innerHTML = '';
    document.getElementById('recommendations-container').style.display = 'none';
    // Ocultar el botón de "Mostrar más" si existe
    const showMoreButton = document.getElementById('show-more-button');
    if (showMoreButton) {
        showMoreButton.style.display = 'none';
    }
});

// ----------------------------
// MOSTRAR RECOMENDACIONES CON BOTÓN "MOSTRAR MÁS"
// ----------------------------

// Variable global para almacenar todas las recomendaciones recibidas
let allRecommendations = [];

/**
 * Renderiza un bloque de recomendaciones a partir de un índice.
 * @param {number} startIndex - Índice desde el cual empezar a renderizar.
 * @param {number} count - Cantidad de recomendaciones a renderizar.
 */
function renderRecommendations(startIndex, count) {
    const animeList = document.getElementById('anime-list');
    allRecommendations.slice(startIndex, startIndex + count).forEach(anime => {
        const listItem = document.createElement('div');
        listItem.classList.add('recommendation');

        listItem.innerHTML = `
            <img src="${anime.main_picture_medium}" alt="${anime.title}" class="anime-image">
            <div class="anime-title">${anime.title}</div>
            <div class="anime-rating">Rating: ${anime.rating}</div>
        `;

        animeList.appendChild(listItem);

        // Al hacer clic en la tarjeta, se muestran los detalles del anime
        listItem.addEventListener('click', function() {
            showSelectedAnime(anime);
        });
    });
}

/**
 * Crea el botón "Mostrar más" si existen recomendaciones pendientes por mostrar.
 * @param {number} currentCount - Número de recomendaciones ya mostradas.
 */
function createShowMoreButton(currentCount) {
    const recommendationsContainer = document.getElementById('recommendations-container');
    let showMoreButton = document.getElementById('show-more-button');

    // Evitar duplicados: eliminar si ya existe
    if (showMoreButton) {
        showMoreButton.remove();
    }

    // Crear el botón solo si quedan recomendaciones pendientes
    if (allRecommendations.length > currentCount) {
        showMoreButton = document.createElement('button');
        showMoreButton.id = 'show-more-button';
        showMoreButton.textContent = 'Mostrar más';

        showMoreButton.addEventListener('click', function() {
            const currentCount = document.querySelectorAll('.recommendation').length;
            renderRecommendations(currentCount, 3);

            // Si se han mostrado todas, eliminar el botón
            if (document.querySelectorAll('.recommendation').length >= allRecommendations.length) {
                showMoreButton.remove();
            }
        });

        recommendationsContainer.appendChild(showMoreButton);
    }
}

// Listener para el envío del formulario: obtención de recomendaciones
document.getElementById('anime-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const animeTitle = document.getElementById('anime-title').value;
    getRecommendations(animeTitle);
});

// ----------------------------
// MOSTRAR DETALLES DE LA RECOMENDACIÓN SELECCIONADA
// ----------------------------
function showSelectedAnime(anime) {
    // Ocultar la lista de recomendaciones y mostrar el contenedor de detalles
    document.getElementById('recommendations-container').style.display = 'none';
    const selectedAnimeContainer = document.getElementById('selected-anime-container');
    selectedAnimeContainer.style.display = 'block';

    document.getElementById('selected-anime-title').textContent = anime.title;
    document.getElementById('selected-anime-image').src = anime.main_picture_medium;
    document.getElementById('selected-anime-rating').textContent = `Rating: ${anime.rating}`;
    document.getElementById('selected-anime-episodes').textContent = `Episodes: ${anime.num_episodes}`;
    document.getElementById('selected-anime-synopsis').textContent = `Description: ${anime.synopsis}`;
}





// ----------------------------
// BOTÓN "BACK" PARA VOLVER A LAS RECOMENDACIONES
// ----------------------------
document.getElementById('back-button').addEventListener('click', function() {
    document.getElementById('selected-anime-container').style.display = 'none';
    document.getElementById('recommendations-container').style.display = 'block';
});
