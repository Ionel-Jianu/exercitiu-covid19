import streamlit as st
st.set_page_config(page_title="Detector COVID-19",page_icon="covid19.jpeg",layout='centered',initial_sidebar_state='auto')

import os
import time

# Pachetele pentru vizualizare
import cv2
from PIL import Image,ImageEnhance
import numpy as np

# Pachetele AI
import tensorflow as tf

def main():
    """ Instrument simplu de detectare a COVID19 pe baza unei poze X-Ray a pieptului. """
    șablon_html="""
    <div style="background-color:blue;padding:10px;">
        <h1 style="color:yellow">Instrument de detectare a COVID19</h1>
    </div>
    """

    st.markdown(șablon_html,unsafe_allow_html=True)
    st.write("Un instrument simplu de diagnoză a COVID19 ce folosește Deep Learning și Streamlit.")
    st.sidebar.image("covid19.jpeg",width=300)

    fișier_încărcat_imagine=st.sidebar.file_uploader("Încărcați o imagine X-Ray (jpg, jpeg sau png)",type=["jpg","jpeg","png"])
    if fișier_încărcat_imagine is not None:
        fișier_imagine=Image.open(fișier_încărcat_imagine)

        if st.sidebar.button("Previzualizare"):
            st.sidebar.image(fișier_imagine,width=300)

        opțiuni_meniu=["Îmbunătățire imagine","Diagnoză","Informații"]
        opțiune_aleasă=st.sidebar.selectbox("Alegeți:",opțiuni_meniu)
        if opțiune_aleasă=="Îmbunătățire imagine":
            st.subheader("Îmbunătățire imagine")
            tip_îmbunătățire=st.sidebar.radio("Tipul îmbunătățirii",["Original","Contrast","Luminozitate"])
            if tip_îmbunătățire=="Original":
                st.write("Imaginea originală")
                st.image(fișier_imagine,width=600,use_container_width=True)
            elif tip_îmbunătățire=="Contrast":
                rată_contrast=st.slider("Contrast",1.0,5.0)
                obiect_imagine=ImageEnhance.Contrast(fișier_imagine)
                imagine_modificată=obiect_imagine.enhance(rată_contrast)
                st.image(imagine_modificată,width=600,use_container_width=True)

            elif tip_îmbunătățire=="Luminozitate":
                rată_luminozitate=st.slider("Luminozitate",1.0,5.0)
                obiect_imagine=ImageEnhance.Brightness(fișier_imagine)
                imagine_modificată=obiect_imagine.enhance(rată_luminozitate)
                st.image(imagine_modificată,width=600,use_container_width=True)

        elif opțiune_aleasă=="Diagnoză":
            st.subheader("Diagnosticare")
            if st.sidebar.button("Diagnoză"):
                matrice_imagine = np.array(fișier_imagine.convert('RGB')) # imagine -> matrice
                imagine_gri = cv2.cvtColor(matrice_imagine, 1) # 0 înseamnă original, 1 înseamnă scală de gri
                imagine_alb_negru = cv2.cvtColor(imagine_gri, cv2.COLOR_BGR2GRAY) # -> în imagine alb-negru
                st.text("Poză X-Ray a pieptului")
                st.image(imagine_alb_negru,width=400,use_container_width=True)
                
                # Pre-procesarea imaginii X-Ray
                IMG_SIZE=(200,200)
                imagine_prelucrată=cv2.equalizeHist(imagine_alb_negru)
                imagine_prelucrată=cv2.resize(imagine_prelucrată,IMG_SIZE)
                imagine_prelucrată=imagine_prelucrată/255 # normalizare

                # redimensionare imagine corespunzător formatului TensorFlow
                imagine_prelucrată=imagine_prelucrată.reshape(1,200,200,1)

                # încărcăm modelul CNN pre-antrenat
                model=tf.keras.models.load_model("./modele/Covid19_CNN_Classifier.h5")

                # diagnostic (Predicție==Clasificare Binară)
                diagnostic_probă=model.predict(imagine_prelucrată)
                diagnostic=np.argmax(diagnostic_probă,axis=1)

                # afișăm progresul predicției
                bară_progres_predicție=st.sidebar.progress(0)
                for procent_completare in range(100):
                    time.sleep(0.05)
                    bară_progres_predicție.progress(procent_completare+1)
                    #
                    # afișăm diagnosticul pe ecran
                    if diagnostic==0:
                        st.sidebar.success("Diagnostic : nu are COVID.")
                    else:
                        st.sidebar.error("Diagnostic: ARE COVID !")
                    st.warning("Această aplicație web este DEMONSTRATIVĂ, diagnosticele sale neavând nici o valoare clinică !")

        else:
            st.subheader("Declinarea răspunderii")
            st.subheader("*Condiții de utilizare*")
            st.write("**Acest instrument este doar o demonstrație despre Rețelele Neuronale Artificiale, diagnosticele sale neavând nici o valoare clinică.**")
            st.write("**Vă rugăm SĂ NU LUAȚI ÎN SERIOS aceste diagnosticuri și NICIODATĂ să nu le considerați valide !**")
            st.subheader("*Informații*")
            st.write("Acest instrument a fost inspirat din următoarele surse :")

    if st.sidebar.button("Despre autor"):
        st.sidebar.subheader("Detector COVID")
        st.sidebar.markdown("creat de [Ionel Jianu](https://www.streamlit.io)")
        st.sidebar.markdown("[ionel.jianu@gmail.com](mailto:ionel.jianu@gmail.com)")
        st.sidebar.text("Drepturi rezervate(2024)")


if __name__=='__main__':
    main()
