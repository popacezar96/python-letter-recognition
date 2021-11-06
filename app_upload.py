from flask import Flask, render_template, request, redirect, flash, url_for

import numpy as np
import cv2 as cv
import tensorflow as tf
import os

from indici import toti_indicii, litere_inalte_de_incredere, litere_mici_de_incredere

from keras.models import load_model

#### Ca sa pot apela model.predict() in mai multe locuri####

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

###########################################################

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model('sigmoid_tf_576_doar_litere.h5')

def fct1(imagine):
    ### Procesare grafica pentru gasirea contururilor si incadrarea lor in dreptunghiuri ###

    ret, img_thresh = cv.threshold(imagine, 180, 255, cv.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    img_thresh = cv.erode(img_thresh, kernel, iterations=1)

    iesire_canny = cv.Canny(img_thresh, 200, 255)

    _, contururi, hier = cv.findContours(iesire_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rectangle = []

    for ctr in contururi:
        rectangle.append(cv.boundingRect(ctr))

    contururi = np.array(rectangle)

    return contururi, img_thresh


def fct2(img_thresh):
    ### Redimensionare si clasificarea imaginii ###
    bucata = cv.resize(img_thresh, (28, 28))

    bucata_4D = bucata.reshape(1, 28, 28, 1)
    bucata_4D = tf.cast(bucata_4D, tf.float32)

    pred = model.predict(bucata_4D)

    pr = pred.argmax()
    litera = list(toti_indicii.keys())[list(toti_indicii.values()).index(pr)]

    return litera[0]


def fct3(contururi, img_thresh):

    ### SORTARE CONTURURI de la STANGA la DREAPTA ###
    if len(contururi) > 1:
        contururi_sortate = contururi[np.argsort(contururi[:, 0])]

        ### Gasirea contururilor din interiorul altor contururi (contururi suplimentare)###

        contururi_gresite = []
        contururi_gresite.clear()

        con = 0
        for i in contururi_sortate:
            if con != 0:
                if contururi_sortate[con - 1][0] <= i[0] <= contururi_sortate[con - 1][0] + \
                        contururi_sortate[con - 1][2] and contururi_sortate[con - 1][1] <= i[1] <= \
                        contururi_sortate[con - 1][1] + contururi_sortate[con - 1][3]:
                    contururi_gresite.append(i)
            con += 1

        ########## Eliminare contururi suplimentare #########

        for x, y, w, h in contururi_sortate:
            for a, b, c, d in contururi_gresite:
                if [x, y, w, h] == [a, b, c, d]:
                    contururi_sortate = np.delete(contururi_sortate,
                                                  np.where((contururi_sortate == [x, y, w, h]).all(axis=1))[
                                                      0][0], axis=0)
        if len(contururi_sortate) > 1:
            ####Cod pt literele i si j individuale####
            vector_i = []
            jumatati = []
            for x, y, w, h in contururi_sortate:
                jumatati.append(x + w / 2)

            cel_mai_inalt = max(i[3] for i in contururi_sortate)

            cont_jum = 0

            for i in contururi_sortate:
                if i[3] * 3 < cel_mai_inalt:
                    if jumatati[cont_jum] - jumatati[cont_jum - 1] > jumatati[cont_jum + 1] - jumatati[
                        cont_jum]:

                        inaltime_sigura_1 = i[3] + contururi_sortate[cont_jum + 1][3] + \
                                            contururi_sortate[cont_jum + 1][1] - i[1] - i[3]

                        if i[0] + i[2] >= contururi_sortate[cont_jum + 1][0] + \
                                contururi_sortate[cont_jum + 1][2]:
                            vector_i.append([i[0], i[1], i[2], inaltime_sigura_1])

                        elif i[0] + i[2] < contururi_sortate[cont_jum + 1][0] + \
                                contururi_sortate[cont_jum + 1][2]:
                            vector_i.append([i[0], i[1], contururi_sortate[cont_jum + 1][0] +
                                             contururi_sortate[cont_jum + 1][2] - i[0], inaltime_sigura_1])

                    elif jumatati[cont_jum] - jumatati[cont_jum - 1] < jumatati[cont_jum + 1] - jumatati[
                        cont_jum]:

                        inaltime_sigura_2 = contururi_sortate[cont_jum - 1][3] + i[3] + \
                                            contururi_sortate[cont_jum - 1][1] - i[1] - i[3]

                        if i[0] + i[2] >= contururi_sortate[cont_jum - 1][0] + \
                                contururi_sortate[cont_jum - 1][2]:
                            vector_i.append([contururi_sortate[cont_jum - 1][0], i[1],
                                             i[0] + i[2] - contururi_sortate[cont_jum - 1][0],
                                             inaltime_sigura_2])

                        elif i[0] + i[2] < contururi_sortate[cont_jum - 1][0] + \
                                contururi_sortate[cont_jum - 1][2]:
                            vector_i.append([contururi_sortate[cont_jum - 1][0], i[1],
                                             contururi_sortate[cont_jum - 1][0] +
                                             contururi_sortate[cont_jum - 1][2] - i[0], inaltime_sigura_2])

                cont_jum += 1

            for x, y, w, h in vector_i:
                for a, b, c, d in contururi_sortate:
                    if x <= a <= x + w:
                        contururi_sortate = np.delete(contururi_sortate, np.where(
                            (contururi_sortate == [a, b, c, d]).all(axis=1))[0][0], axis=0)

            for i in vector_i:
                contururi_sortate = np.append(contururi_sortate, [i], axis=0)

            return contururi_sortate

        else:
            return fct2(img_thresh)

    else:
        return fct2(img_thresh)

def fct4(rezultat_fct_3,img_thresh):
    ######## Prelucrare grafica pentru cuvinte ########

    contururi_sortate = rezultat_fct_3[np.argsort(rezultat_fct_3[:, 0])]

    ############# CALCULAREA DISTANTEI DINTRE LITERE ###############

    distanta = []
    cont_dist = 0

    for spatiu in range(0,len(contururi_sortate)):
        if cont_dist + 1 < len(contururi_sortate):
            urmat_litera = contururi_sortate[cont_dist + 1][0]
            prima_litea = contururi_sortate[cont_dist][0] + \
                          contururi_sortate[cont_dist][2]

            distanta.append(urmat_litera - prima_litea)
        cont_dist += 1

    distanta_adaptata = min(distanta)

    #################################################################

    dictionar_litere = {}
    dict_litere_mari = {}
    dict_litere_mici = {}

    nr_dict = 0

    for x, y, w, h in contururi_sortate:

        grosime = x + w + distanta_adaptata
        inaltime = y + h + distanta_adaptata

        bucata = np.copy(
            img_thresh[y - distanta_adaptata:inaltime, x - distanta_adaptata:grosime])
        bucata = cv.resize(bucata, (28, 28))

        # Modificare pt tf 2.0

        bucata_4D = bucata.reshape(1, 28, 28, 1)
        bucata_4D = tf.cast(bucata_4D, tf.float32)

        pred = model.predict(bucata_4D)

        ####################

        pr = pred.argmax()
        litera = list(toti_indicii.keys())[list(toti_indicii.values()).index(pr)]

        ##### Dictionarele nu pot avea doua key identice asa ca adaug un contor la fiecare key ###

        if litera in dictionar_litere:

            litera = litera + str(nr_dict)
            dictionar_litere.update({litera: y})
            nr_dict += 1

        else:
            dictionar_litere.update({litera: y})

            #######################################################

        if litera[0:5] in litere_inalte_de_incredere:

            dict_litere_mari.update({litera: y})

        elif litera[0:5] in litere_mici_de_incredere:
            dict_litere_mici.update({litera: y})

    vector_litere_final = []

    ### Daca avem sigur litere inalte si litere mici ###
    if dict_litere_mici and dict_litere_mari:

        min_lit_mica = max(dict_litere_mici[i] for i in dict_litere_mici)

        max_lit_mare = min(dict_litere_mari[i] for i in dict_litere_mari)

        for litera in dictionar_litere:

            if litera[0:5] not in litere_mici_de_incredere and litera[
                                                               0:5] not in litere_inalte_de_incredere:

                if min_lit_mica - dictionar_litere[litera] < dictionar_litere[litera] - max_lit_mare:
                    vector_litere_final.append(litera[0].lower())

                elif min_lit_mica - dictionar_litere[litera] > dictionar_litere[litera] - max_lit_mare:
                    vector_litere_final.append(litera[0].upper())

                else:
                    vector_litere_final.append(litera[0])

            else:
                vector_litere_final.append(litera[0])

    ### Daca avem sigur litere mici ###
    elif dict_litere_mici and not dict_litere_mari:

        max_lit_mica = min(dict_litere_mici[i] for i in dict_litere_mici)

        for litera in dictionar_litere:
            if litera[0:5] not in litere_mici_de_incredere and litera[
                                                               0:5] not in litere_inalte_de_incredere:

                if dictionar_litere[litera] + 3 >= max_lit_mica:
                    vector_litere_final.append(litera[0].lower())

                else:
                    vector_litere_final.append(litera[0].upper())
            else:
                vector_litere_final.append(litera[0])

    ### Daca avem sigur litere mari ###
    elif not dict_litere_mici and dict_litere_mari:

        min_lit_mare = max(dict_litere_mari[i] for i in dict_litere_mari)

        for litera in dictionar_litere:
            if litera[0:5] not in litere_mici_de_incredere and litera[
                                                               0:5] not in litere_inalte_de_incredere:

                if dictionar_litere[litera] - 3 <= min_lit_mare:
                    vector_litere_final.append(litera[0].upper())
                else:
                    vector_litere_final.append(litera[0].lower())
            else:
                vector_litere_final.append(litera[0])

    ### Nu stim daca literele sunt mici sau mari ###
    else:

        cea_mai_scunda = max(dictionar_litere[i] for i in dictionar_litere)
        cea_mai_inalta = min(dictionar_litere[i] for i in dictionar_litere)

        diferenta = cea_mai_scunda - cea_mai_inalta

        if diferenta >= 5:
            ### Avem litere mici si mari ###
            for litera in dictionar_litere:

                if cea_mai_scunda - dictionar_litere[litera] < dictionar_litere[
                    litera] - cea_mai_inalta:
                    vector_litere_final.append(litera[0].lower())

                elif cea_mai_scunda - dictionar_litere[litera] > dictionar_litere[
                    litera] - cea_mai_inalta:
                    vector_litere_final.append(litera[0].upper())

                else:
                    vector_litere_final.append(litera[0])
        else:
            ### Avem doar litere mari sau doar litere mici ###
            for litera in dictionar_litere:
                ### Le fac pe toate mici mai probabil sa apara doua litere mici decat doua litere mari intr-un cuvant ###
                vector_litere_final.append(litera[0].lower())

    vector_litere_final = ''.join(vector_litere_final) + ' '

    return vector_litere_final

@app.route("/")
@app.route("/upload", methods=['GET', 'POST'])
def upload():

    return render_template("upload.html")


@app.route("/litera", methods=['GET', 'POST'])
def litera():

    target = os.path.join(APP_ROOT, 'static/')

    if request.method == 'POST':

        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("file"):

            poza_litera = file.filename
            if poza_litera:
                destination = "/".join([target, poza_litera])

                file.save(destination)

                imagine = cv.imread(destination, 0)

                contururi, img_thresh= fct1(imagine)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                if isinstance(rezultat_fct_3, str):
                    litera = rezultat_fct_3
                    return render_template("litera.html", litera=litera, poza_litera=poza_litera)
                else:
                    prea_multe = True
                    litera = ''
                    return render_template("litera.html", litera=litera, poza_litera=poza_litera, prea_multe=prea_multe)

            else:
                return redirect(request.url)
    else:
        pass

    return render_template('litera.html', litera = '', poza_litera = '')

@app.route("/cuvant", methods=['GET', 'POST'])
def cuvant():
    target = os.path.join(APP_ROOT, 'static/')

    if request.method == 'POST':

        if not os.path.isdir(target):
            os.mkdir(target)

        for file1 in request.files.getlist("file"):

            poza_cuvant = file1.filename
            if poza_cuvant:
                destination = "/".join([target, poza_cuvant])
                imagine = cv.imread(destination, 0)

                contururi, img_thresh = fct1(imagine)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                if isinstance(rezultat_fct_3, str):
                    cuvant = ''
                    prea_putine = True
                    return render_template("cuvant.html", cuvant=cuvant, poza_cuvant=poza_cuvant, prea_putine=prea_putine)
                else:

                    cuvant = fct4(rezultat_fct_3,img_thresh)
                    return render_template("cuvant.html", cuvant=cuvant, poza_cuvant=poza_cuvant)
            else:
                return redirect(request.url)

    else:
        pass
    return render_template('cuvant.html', cuvant='', poza_cuvant='')

@app.route("/propozitie", methods=['GET', 'POST'])
def propozitie():
    target = os.path.join(APP_ROOT, 'static/')

    if request.method == 'POST':

        if not os.path.isdir(target):
            os.mkdir(target)

        for file1 in request.files.getlist("file"):

            poza_propozitie = file1.filename
            if poza_propozitie:
                destination = "/".join([target, poza_propozitie])
                file1.save(destination)

                imagine = cv.imread(destination, 0)

                imagine1 = np.copy(imagine[60:210, 120:390])
                imagine1 = cv.resize(imagine1, (186, 69))
                contururi, img_thresh = fct1(imagine1)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                rez1 = fct4(rezultat_fct_3, img_thresh)

                imagine2 = np.copy(imagine[60:210, 460:690])
                imagine2 = cv.resize(imagine2, (97, 61))
                contururi, img_thresh = fct1(imagine2)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                rez2 = fct4(rezultat_fct_3, img_thresh)

                imagine3 = np.copy(imagine[265:370, 125:440])
                imagine3 = cv.resize(imagine3, (137, 61))
                contururi, img_thresh = fct1(imagine3)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                rez3 = fct4(rezultat_fct_3, img_thresh)

                imagine4 = np.copy(imagine[230:370, 490:830])
                imagine4 = cv.resize(imagine4, (186, 69))
                contururi, img_thresh = fct1(imagine4)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                rez4 = fct4(rezultat_fct_3, img_thresh)

                imagine5 = np.copy(imagine[405:550, 130:430])
                imagine5 = cv.resize(imagine5, (137, 61))
                contururi, img_thresh = fct1(imagine5)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                rez5 = fct4(rezultat_fct_3, img_thresh)

                imagine6 = np.copy(imagine[440:550, 480:690])
                imagine6 = cv.resize(imagine6, (97, 61))
                contururi, img_thresh = fct1(imagine6)
                rezultat_fct_3 = fct3(contururi, img_thresh)

                rez6 = fct4(rezultat_fct_3, img_thresh)

                propozitie =  rez1 + rez2 + rez3 + rez4 + rez5 + rez6 + '.'

                return render_template("propozitie.html", propozitie=propozitie, poza_propozitie=poza_propozitie)
            else:
                return redirect(request.url)
    else:
        pass

    return render_template('propozitie.html', propozitie='',poza_propozitie='')

if __name__ == "__main__":
    app.run(debug=True)