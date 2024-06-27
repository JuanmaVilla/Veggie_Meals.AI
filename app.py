from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import openai

def classify_fruit(img):
    np.set_printoptions(suppress=True)
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name.strip(), confidence_score

# OpenAI API
from openai import OpenAI

client = OpenAI(api_key='sk-proj-OgcX2gKKz9TDe1BEdQohT3BlbkFJM6Jm3uEYXuSbs2v7cYiY')

def generate_recipe(ingredients, meal_type, temperature, include_promoted_ingredients):
    MODEL = "gpt-3.5-turbo"
    if include_promoted_ingredients:
        user_message = (
            f"Vas a crear recetas para tu libro de cocina vegetariana. Ahora tienes la misión de crear o recomendar una "
            f"receta de {meal_type} que use los siguientes ingredientes: {ingredients}. Primero, repíteme los ingredientes "
            f"que te pasé en la lista. Luego, clasifica la cantidad de cada ingrediente que debemos usar para la receta. "
            f"Tienes la posibilidad de usar todos los ingredientes de la lista o también puedes omitir algunos. En caso "
            f"que omitas algún ingrediente de la lista proporcionada anteriormente, tienes que nombrar qué ingrediente omitiste. "
            f"Para la receta, puedes utilizar un máximo de 8 ingredientes que no estén en la lista anterior. En caso que para "
            f"la receta utilices ingredientes que no están en la lista, nombra qué ingredientes son. La receta tiene que estar "
            f"en un formato como de libro de cocina. Tiene que ser amigable e intuitivo. Muchas gracias :)"
        )
    else:
        user_message = (
            f"Vas a crear recetas para tu libro de cocina vegetariana. Ahora tienes la misión de crear o recomendar una "
            f"receta de {meal_type} que use únicamente los siguientes ingredientes: {ingredients}. Primero, repíteme los ingredientes "
            f"que te pasé en la lista. Luego, clasifica la cantidad de cada ingrediente que debemos usar para la receta. "
            f"No puedes omitir ningún ingrediente de la lista ni agregar ingredientes adicionales. La receta tiene que estar "
            f"en un formato como de libro de cocina. Tiene que ser amigable e intuitivo. Muchas gracias :)"
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Eres una chef de cocina vegetariana que escribe un libro de cocina."},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

# Streamlit App
st.set_page_config(layout='wide')

st.image('logo.png', caption=None, width=300, use_column_width=None)
st.title("Recetas vegetarianas con AI! 😉")
st.subheader("Crea recetas para cualquier momento del día!")
st.subheader("¡Toma fotos de los ingredientes que tengas en tu cocina y los identificamos! ")
st.subheader("Luego, podrás crear recetas, para cualquier momento del día, con los ingredientes que subiste!🍴 Elige entre desayunos, postres, almuerzos o cenas. Tenemos lo mejor para vos!")

# Lista de ingredientes en sesión
if 'ingredientes' not in st.session_state:
    st.session_state.ingredientes = []

# Lista de recetas generadas
if 'recetas' not in st.session_state:
    st.session_state.recetas = []

# Selección de opción
option = st.selectbox(
    'Selecciona una opción:',
    ('Selecciona una opción', 'Tomar una foto con la cámara', 'Subir una imagen desde el dispositivo')
)

def procesar_imagen(image_file):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("Imagen cargada")
        st.image(image_file, use_column_width=True)

    with col2:
        st.info("Resultado")
        image = Image.open(image_file)

        with st.spinner('Analizando imagen...'):
            label, confidence_score = classify_fruit(image)
            if confidence_score < 0.8:
                st.error("La foto ingresada no se pudo identificar. Por favor, vuelva a tomar la foto.")
            else:
                label_description = label.split(maxsplit=1)[1]  # Divide la etiqueta por el primer espacio y toma el segundo elemento
                label2 = label_description  # Guarda la descripción en label2

                st.success(label2)  # Muestra la etiqueta sin el número

                # Añadir el ingrediente a la lista en sesión
                st.session_state.ingredientes.append(label2)

if option == 'Tomar una foto con la cámara':
    st.warning("Haz clic en el botón para permitir el acceso a la cámara.")

    # Redimensionar la cámara utilizando columnas
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        img_file_buffer = st.camera_input("Toma una foto")

    if img_file_buffer is not None:
        procesar_imagen(img_file_buffer)

elif option == 'Subir una imagen desde el dispositivo':
    input_img = st.file_uploader("Elegir imagen", type=['jpg', 'png', 'jpeg'])

    if input_img is not None:
        if st.button("Determinar el ingrediente"):
            procesar_imagen(input_img)

# Filtrar Ingredientes
st.subheader("Ingredientes identificados:")
for i, ingrediente in enumerate(st.session_state.ingredientes):
    col1, col2 = st.columns([4, 1])
    col1.write(ingrediente)
    if col2.button("❌", key=f"eliminar_{i}"):
        st.session_state.ingredientes.pop(i)
        st.experimental_rerun()

# Slider para ajustar la creatividad de la receta
st.subheader("Ajusta la creatividad de la receta:")
temperature = st.slider("Creatividad", 0.0, 1.0, 0.0)

# Checkbox para incluir ingredientes promocionados
st.subheader("¿Incluir ingredientes promocionados?")
include_promoted_ingredients = st.checkbox("Sí, quiero incluir ingredientes promocionados.")

# Mostrar los botones de generación de recetas solo si hay ingredientes
if st.session_state.ingredientes:
    st.subheader("Genera tu receta:")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Generar desayuno"):
        meal_type = "desayuno"
        with st.spinner('Generando receta de desayuno...'):
            recipe = generate_recipe(st.session_state.ingredientes, meal_type, temperature, include_promoted_ingredients)
            st.session_state.recetas.append((meal_type, recipe, list(st.session_state.ingredientes)))
            st.success(recipe)

    if col2.button("Generar postre"):
        meal_type = "postre"
        with st.spinner('Generando receta de postre...'):
            recipe = generate_recipe(st.session_state.ingredientes, meal_type, temperature, include_promoted_ingredients)
            st.session_state.recetas.append((meal_type, recipe, list(st.session_state.ingredientes)))
            st.success(recipe)

    if col3.button("Generar almuerzo"):
        meal_type = "almuerzo"
        with st.spinner('Generando receta de almuerzo...'):
            recipe = generate_recipe(st.session_state.ingredientes, meal_type, temperature, include_promoted_ingredients)
            st.session_state.recetas.append((meal_type, recipe, list(st.session_state.ingredientes)))
            st.success(recipe)

    if col4.button("Generar cena"):
        meal_type = "cena"
        with st.spinner('Generando receta de cena...'):
            recipe = generate_recipe(st.session_state.ingredientes, meal_type, temperature, include_promoted_ingredients)
            st.session_state.recetas.append((meal_type, recipe, list(st.session_state.ingredientes)))
            st.success(recipe)

# Mostrar lista de recetas anteriores
st.subheader("Recetas anteriores:")
if st.session_state.recetas:
    for i, (meal_type, receta, ingredientes_usados) in enumerate(st.session_state.recetas):
        with st.expander(f"Receta {i+1}: {meal_type.capitalize()} con {', '.join(ingredientes_usados)}"):
            st.write(receta)
else:
    st.info("Aún no has generado ninguna receta.")

# CSS para mejorar la estética
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }

    .stButton> {
        background-color: #FFA07A;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }

    .stInfo, .stSuccess, .stWarning, .stError {
        background-color: #F5F5DC;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)