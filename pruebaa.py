from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
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

# import the OpenAI Python library for calling the OpenAI API
from openai import OpenAI

client = OpenAI(api_key='sk-proj-W8VQBjogkecViLxwitF4T3BlbkFJD0vFIcnyWKCYfugdoWb5')

def generate_recipe(ingredients):

    MODEL = "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Eres una chef de cocina vegetariana que escribe un libro de cocina."},
            {"role": "user", "content": f"Vas a crear recetas para tu libro de cocina vegetariana. Ahora tienes la mision de crear o recomendar una receta que use los siguientes ingredientes: {ingredients}. Primero, repiteme los ingredientes que te pase en la lista. Luego, clasifica la cantidad de cada ingrediente que debemos usar para la receta. Tienes la posibilidad de usar todos los ingredientes de la lista o tambien puede omitir algunos. En caso que omitas algun ingrediente de la lista proporsionada anteriormente, tienes que nombrar que ingrediente omitiste. Para la receta, puedes utilizar un maximo de 8 ingredientes que no esten en la lista anterior. En caso que para la receta utilices ingredientes que no estan en la lista, nombra que ingredientes son. La reseta tiene que estar en un formato como de libro de cocina. Tiene que ser amigable y intuitivo. Muchas gracias :)"},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# Streamlit App
st.set_page_config(layout='wide')

st.image('logo.png', caption=None, width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.title("Recetas vegetarianas con AI! 😉")
st.subheader("¡Toma fotos de los ingredientes que tengas en tu cocina y los identificamos!")
st.subheader("Luego, podrás crear recetas con los ingredientes que subiste!🍴")

# Lista de ingredientes en sesión
if 'ingredientes' not in st.session_state:
    st.session_state.ingredientes = []

# Selección de opción
option = st.selectbox(
    'Selecciona una opción:',
    ('Selecciona una opción', 'Tomar una foto con la cámara', 'Subir una imagen desde el dispositivo')
)

def procesar_imagen(image_file):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.info("Imagen cargada")
        st.image(image_file, use_column_width=True)

    with col2:
        st.info("Resultado")
        image = Image.open(image_file)

        with st.spinner('Analizando imagen...'):
            label, confidence_score = classify_fruit(image)
            label_description = label.split(maxsplit=1)[1]  # Divide la etiqueta por el primer espacio y toma el segundo elemento
            label2 = label_description  # Guarda la descripción en label2

            st.success(label2)  # Muestra la etiqueta sin el número

            # Añadir el ingrediente a la lista en sesión
            st.session_state.ingredientes.append(label2)

    with col3:
        st.info("Ingredientes añadidos:")
        for ingrediente in st.session_state.ingredientes:
            st.info(ingrediente)


if option == 'Tomar una foto con la cámara':
    st.warning("Haz clic en el botón para permitir el acceso a la cámara.")


    img_file_buffer = st.camera_input("Toma una foto")
        
    if img_file_buffer is not None:
        procesar_imagen(img_file_buffer)

elif option == 'Subir una imagen desde el dispositivo':
    input_img = st.file_uploader("Elegir imagen", type=['jpg', 'png', 'jpeg'])

    if input_img is not None:
        if st.button("Determinar el ingrediente"):
            procesar_imagen(input_img)
        

if st.button("Generar receta con los ingredientes"):
    if st.session_state.ingredientes:
        with st.spinner('Generando receta...'):
            recipe = generate_recipe(st.session_state.ingredientes)
            st.success(recipe)
    else:
        st.warning("No has añadido ningún ingrediente todavía 🥹")




