from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st 
from openai import OpenAI



def classify_fruit(img):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    return class_name, confidence_score


def generate_recipe(label):


    client = OpenAI(api_key="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")

    

    response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt= f"Sos un asistente experto en cocina con frutas y tenes que recomendar solo 3 ideas de comida para hacer con {label}. Puede ser algo comestible o bebible, considerando si la fruta está buena o mala. No hace falta que expliques las recetas, solo una lista con 3 ideas",
    temperature=0.5,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].text



# Streamlit App

st.set_page_config(layout='wide')

st.title("VEGGIE MEALS!")

st.subheader("""Saca fotos de los ingredientes que tengas en la cocina y te los clasificaremos.""")
st.subheader("""Luego podras crear recetas con tus ingredientes disponibles!🍴""")
input_img = st.file_uploader("Elegir imagen", type=['jpg', 'png', 'jpeg'])

ingredientes = []

if input_img is not None:
    if st.button("Determinar el ingrediente"):
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.info("Imagen cargada")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Resultado")
            image_file = Image.open(input_img)

            with st.spinner('Analizando imagen...'):
                label, confidence_score = classify_fruit(image_file)

                # Extraer el nombre de la etiqueta sin el número
                label_description = label.split(maxsplit=1)[1]  # Divide la etiqueta por el primer espacio y toma el segundo elemento
                label2 = label_description  # Guarda la descripción en label2

                st.success(label2)  # Muestra la etiqueta sin el número
                ingredientes.append(label2)

            
        with col3:
                st.info("Ingredientes añadidos:")
                st.info(ingredientes)
                result = generate_recipe(label2)
                st.success(result)