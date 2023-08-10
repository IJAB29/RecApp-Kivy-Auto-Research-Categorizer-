import ast
import os
import string
import tempfile
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from kivy.app import App
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import StringProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from plyer import filechooser
from pdf2image import convert_from_path
import pytesseract
import natsort
from PIL import Image
import pickle
from kivy.core.window import Window

size = (350, 600)
Window.size = size

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
from gensim import corpora, models
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
custom = ["also", "would", "will"]
stop_words.extend(custom)

poppler_path = r"C:\Program Files\poppler-22.04.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

PICKLE_FOLDER = r"C:\Users\USER\Thesis_env\pickle"

PRED_LBL_FONT_SIZE = dp(20)

RESULTS = {"Names": [],
           "Predictions": [],
           "Confidence": [],
           "Topics": []}


class FileTypeWindow(Screen):
    # def helpImages(self):
    #     popup = Popup()
    #     popup.title = "Use Images Help"
    #     popup.content = Label(
    #         text="Import an image or set of images from research papers to classify.\n"
    #              "It will return the predicted category of each image or set of image that you chose.")
    #     popup.size_hint = (None, None)
    #     popup.size = (size[0] - 50, size[1] - 100)
    #     popup.open()
    #
    # def helpFolders(self):
    #     popup = Popup()
    #     popup.title = "Use Folders of Images Help"
    #     popup.content = Label(
    #         text="Import a folder containing folders of images taken from research papers.\n"
    #              "It will return the predicted category of the extracted images from each folder.")
    #     popup.size_hint = (None, None)
    #     popup.size = (size[0] - 50, size[1] - 100)
    #     popup.open()
    #
    # def helpPdfs(self):
    #     popup = Popup()
    #     popup.title = "Use PDFs Help"
    #     popup.content = Label(
    #         text="Import a PDF or multiple PDFs of research papers to classify.\n"
    #              "It will return the predicted category of each PDF that you chose.")
    #     popup.size_hint = (None, None)
    #     popup.size = (size[0] - 50, size[1] - 100)
    #     popup.open()
    pass


class DatasetWindow(Screen):
    clf = None
    vec = None

    def loadBG(self):
        # self.clf = pickle.load(open(os.path.join(PICKLE_FOLDER, "BACKGROUND SQRT TF IGM MultinomialNB.pkl"), "rb"))
        # self.vec = pickle.load(open(os.path.join(PICKLE_FOLDER, "Dictionary BACKGROUND SQRT TF IGM.pkl"), "rb"))
        self.clf = pickle.load(open(os.path.join(PICKLE_FOLDER, "BACKGROUND SQRT TF MONO LinearSVC.pkl"), "rb"))
        self.vec = pickle.load(open(os.path.join(PICKLE_FOLDER, "Dictionary BACKGROUND SQRT TF MONO.pkl"), "rb"))

    def loadAB(self):
        # self.clf = pickle.load(open(os.path.join(PICKLE_FOLDER, "ABSTRACT SQRT TF IGM MultinomialNB.pkl"), "rb"))
        # self.vec = pickle.load(open(os.path.join(PICKLE_FOLDER, "Dictionary ABSTRACT SQRT TF IGM.pkl"), "rb"))
        self.clf = pickle.load(open(os.path.join(PICKLE_FOLDER, "ABSTRACT SQRT TF MONO LinearSVC.pkl"), "rb"))
        self.vec = pickle.load(open(os.path.join(PICKLE_FOLDER, "Dictionary ABSTRACT SQRT TF MONO.pkl"), "rb"))

    def help(self):
        popup = Popup()
        popup.title = "Help"
        popup.content = Label(
            text="Hello, welcome to our Automatic Research Paper Classifier.\n\n\n"
                 "Please choose which dataset you wish to use during text classification.\n\n"
                 "It is recommended that the dataset you will choose should match with the section of the research paper that you wish to classify.\n"
                 "i.e. Use the Abstract Dataset to classify the Abstract of the Study")
        popup.size_hint = (None, None)
        popup.size = (size[0] - 50, size[1] - 100)
        popup.open()


class ImageClfWindow(Screen):
    img_display = StringProperty("")

    def importImages(self):
        images = filechooser.open_file(multiple=True, filters=["*.jpg", "*.jpeg", "*.png"])
        self.img_display = self.ids.txt_input_image_paths.text
        if images:
            self.img_display += f"{images}, "

    def onPressClassify(self):
        clf = self.manager.get_screen("dataset_window").clf
        vec = self.manager.get_screen("dataset_window").vec
        self.img_display = self.ids.txt_input_image_paths.text

        if self.img_display:
            try:
                eval(self.img_display)
            except SyntaxError as err:
                title = f"Syntax Error: {err.msg}"
                content = err.text
                kivyException(title, content)
            else:
                items = ast.literal_eval(self.img_display)
                for i in range(0, len(items)):
                    try:
                        image_list = items[i]
                        if len(image_list) > 0:
                            converted_text = ""
                            for image_path in image_list:
                                ocr = applyOCR(image_path)
                                converted_text += ocr
                            if converted_text:
                                prediction = classify(converted_text, clf, vec)
                                confidence = getConfidence(converted_text, clf, vec)
                                topics = Lda().getTopics(converted_text, 1, 10)
                                storeResults(i, prediction, confidence, topics)
                                category = f"{i}: {prediction} {confidence}\n"
                                displayPredictions(category, self.ids.image_predictions)

                    except Exception as e:
                        title = getattr(e, 'message', repr(e))
                        content = getattr(e, 'message', str(e))
                        kivyException(title, content)

                self.img_display = ""

    def helpImgClfWin(self):
        popup = Popup()
        popup.title = "Procedure"
        popup.content = Label(
            text='You may choose one (1) or multiple images at a time for this input method.\n'
                 'Please note that the program will treat each new input as an independent entity from the previous ones.\n\n'
                 'Step 1: Press the "Choose Images" button to select and import an image or set of images.\n'
                 'Step 2: Press the "Classify" button to predict the category of the inputs and wait for the results.\n\n\n\n'
                 'NOTE: If you wish to remove an input, please delete the complete path from the text input to avoid errors.'
        )
        popup.size_hint = (None, None)
        popup.size = (size[0] - 50, size[1] - 100)
        popup.open()


class FolderClfWindow(Screen):
    dir_display = StringProperty("")

    def uploadFiles(self):
        path = filechooser.choose_dir()
        self.dir_display = self.ids.file_paths.text
        if path:
            self.dir_display = path[0]

    def onButtonPress(self):
        clf = self.manager.get_screen("dataset_window").clf
        vec = self.manager.get_screen("dataset_window").vec
        self.dir_display = self.ids.file_paths.text

        try:
            if self.dir_display:
                path = self.dir_display.split(",")[0]
                dirs = os.listdir(path)

                for filepath in dirs:
                    filepath = os.path.join(path, filepath.strip())
                    if filepath:
                        files = natsort.natsorted(os.listdir(filepath))
                        converted_text = ""
                        for file in files:
                            ocr = applyOCR(os.path.join(filepath, file))
                            converted_text += ocr
                        if converted_text:
                            prediction = classify(converted_text, clf, vec)
                            confidence = getConfidence(converted_text, clf, vec)
                            folder_name = os.path.basename(filepath)
                            topics = Lda().getTopics(converted_text, 1, 10)
                            storeResults(folder_name, prediction, confidence, topics)
                            category = f"{folder_name}: {prediction} {confidence}"
                            displayPredictions(category, self.ids.folder_predictions)

                self.dir_display = ""
        except Exception as e:
            title = getattr(e, 'message', repr(e))
            content = getattr(e, 'message', str(e))
            kivyException(title, content)

    def helpFolClfWin(self):
        popup = Popup()
        popup.title = "Procedure"
        popup.content = Label(
            text='You may only choose one (1) folder for this input method.\n\n'
                 'Step 1: Press the "Choose Directory" button to select a folder containing the folders of images from research papers.\n'
                 'Step 2: Press the "Classify" button to predict the category of the inputs and wait for the results.\n\n\n\n'
                 'NOTE: If you wish to remove an input, please delete the complete path from the text input to avoid errors.'
        )
        popup.size_hint = (None, None)
        popup.size = (size[0] - 50, size[1] - 100)
        popup.open()


class PdfClfWindow(Screen):
    file_display = StringProperty("")

    def openPdf(self):
        pdfs = filechooser.open_file(multiple=True, filters=["*.pdf"])
        self.file_display = self.ids.pdf_files.text
        if pdfs:
            for pdf in pdfs:
                if pdf not in self.file_display.split(","):
                    self.file_display += f"{pdf}, "

    def onPressClassify(self):
        clf = self.manager.get_screen("dataset_window").clf
        vec = self.manager.get_screen("dataset_window").vec
        self.file_display = self.ids.pdf_files.text

        if self.file_display:
            pdfs_to_convert = self.file_display.split(",")

            for pdf_path in pdfs_to_convert:
                try:
                    pdf_path = pdf_path.strip()
                    if pdf_path:
                        converted_text = extractPdfText(pdf_path)
                        if not converted_text:
                            converted_text = pdfToImage(pdf_path)
                        prediction = classify(converted_text, clf, vec)
                        confidence = getConfidence(converted_text, clf, vec)
                        filename = os.path.basename(pdf_path)
                        topics = Lda().getTopics(converted_text, 1, 10)
                        storeResults(filename, prediction, confidence, topics)
                        category = f"{filename}: {prediction} {confidence}"
                        displayPredictions(category, self.ids.pdf_predictions)

                except Exception as e:
                    title = getattr(e, 'message', repr(e))
                    content = getattr(e, 'message', str(e))
                    kivyException(title, content)

            self.file_display = ""

    def helpPdfClfWin(self):
        popup = Popup()
        popup.title = "Procedure"
        popup.content = Label(
            text='You may choose multiple pdf files at a time for this input method.\n\n'
                 'Step 1: Press the "Choose PDF Files" button to select and import pdf files.\n'
                 'Step 2: Press the "Classify" button to predict the category of the inputs and wait for the results.\n\n\n\n'
                 'NOTE: If you wish to remove an input, please delete the complete path from the text input to avoid errors.'
        )
        popup.size_hint = (None, None)
        popup.size = (size[0] - 50, size[1] - 100)
        popup.open()


# class PdfClfWindow(Screen):
#     categories = StringProperty("")
#     file_display = StringProperty("")
#     pdf_file = ObjectProperty()
#
#     def openPdf(self):
#         pdf = filechooser.open_file(filters=["*.pdf"])[0]
#         self.file_display += f"{pdf}, "
#         self.pdf_file = pdf
#
#     def onPressClassify(self):
#         clf = self.manager.get_screen("dataset_window").clf
#         vec = self.manager.get_screen("dataset_window").vec
#         self.file_display = self.ids.pdf_files.text
#         pdfs_to_convert = self.file_display.split(",")
#         file_names = [os.path.basename(pdf).split(".")[0] for pdf in pdfs_to_convert if pdf]
#         file_path_list = os.listdir(PDF2IMG_FOLDER)
#         categories = ""
#         for filepath in file_path_list:
#             file_list = natsort.natsorted(os.listdir(f"{PDF2IMG_FOLDER}/{filepath}"))
#             converted_text = ""
#             if filepath in file_names:
#                 for file in file_list:
#                     ocr = applyOCR(os.path.join(f"{PDF2IMG_FOLDER}/{filepath}", file))
#                     converted_text += ocr
#             if converted_text:
#                 prediction = classify(converted_text, clf, vec)
#                 confidence = getConfidence(converted_text, clf, vec)
#                 filename = os.path.basename(filepath)
#                 categories += f"{filename}: {prediction} {confidence}\n"
#         self.file_display = ""
#         self.categories = categories
#         self.deleteFiles()
#
#     def deleteFiles(self):
#         for file in os.listdir(PDF2IMG_FOLDER):
#             fullpath = os.path.join(PDF2IMG_FOLDER, file)
#             shutil.rmtree(fullpath)
#
#     def helpPdfClfWin(self):
#         pass
#
#
# class PageSelectWindow(Screen):
#     def __init__(self, **kw):
#         super().__init__(**kw)
#         self.images = None
#         self.chosen_pages = None
#         self.pdf = None
#
#     def on_enter(self, *args):
#         self.pdf = self.manager.get_screen("pdf_clf_window").pdf
#         self.createWidgets()
#
#     def createWidgets(self):
#         pdf = self.pdf
#         self.images = convert_from_path(pdf_path=pdf, poppler_path=poppler_path)
#         with tempfile.TemporaryDirectory() as temp_dir:
#             for i in range(len(self.images)):
#                 self.images[i].save(f"{temp_dir}/{i}.jpg", "JPEG")
#             image_list = natsort.natsorted(os.listdir(temp_dir))
#             for i in range(1, len(image_list)):
#                 # height = dp(1280)
#                 # width = dp(720)
#                 image = KivyImg(source=f"{temp_dir}/{image_list[i-1]}",
#                                 size_hint=(None, None),
#                                 size=size)
#                 label = Label(text=f"{i}")
#                 self.ids.container.add_widget(image)
#                 self.ids.container.add_widget(label)
#
#     def savePages(self):
#         self.chosen_pages = self.ids.chosen_pages.text.split(",")
#         folder_name = os.path.basename(self.pdf).split(".")[0]
#         converted_imgs = f"{PDF2IMG_FOLDER}/{folder_name}"
#         if not os.path.exists(converted_imgs):
#             os.makedirs(converted_imgs)
#         for i in self.chosen_pages:
#             i = int(i)
#             self.images[i - 1].save(f"{converted_imgs}/Page {i}.jpg", "JPEG")
#
#         self.ids.chosen_pages.text = ""
#         self.ids.container.clear_widgets()


class WindowManager(ScreenManager):
    pass


def storeResults(name, pred, conf, topics):
    RESULTS["Names"].append(name)
    RESULTS["Predictions"].append(pred)
    RESULTS["Confidence"].append(conf)
    RESULTS["Topics"].append(topics)


def displayPredictions(text, container):
    label = Label()
    label.text = text
    label.font_size = PRED_LBL_FONT_SIZE
    label.size_hint_y = None
    container.add_widget(label)


def kivyException(title, message):
    popup = Popup()
    popup.title = title
    popup.content = Label(text=message)
    popup.size_hint = (None, None)
    popup.size = (size[0] - 50, size[1] - 100)
    popup.open()


def pdfToImage(pdf):
    images = convert_from_path(pdf_path=pdf, poppler_path=poppler_path)
    converted_text = ""
    with tempfile.TemporaryDirectory() as temp:
        for i in range(len(images)):
            images[i].save(f"{temp}/Page {i}.jpg", "JPEG")

        image_list = natsort.natsorted(os.listdir(temp))
        for img in image_list:
            ocr = pytesseract.image_to_string(Image.open(os.path.join(temp, img)))
            converted_text += ocr

    return converted_text


def applyOCR(filepath):
    return pytesseract.image_to_string(Image.open(filepath))


def classify(text, clf, vec):
    return clf.predict(vec.transform([text]))[0]


def getConfidence(text, clf, vec):
    # percentages = clf.predict_proba(vec.transform([text]))[0]
    # confidence = np.sort(percentages)[-1] * 100
    percentages = clf._predict_proba_lr(vec.transform([text]))[0]
    confidence = np.sort(percentages)[-1] * 100
    confidence = format(confidence, '.2f')
    return f"{confidence}%"


def extractPdfText(pdf):
    extracted_text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()

    return extracted_text


class Lda:

    def __init__(self):
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.tokens = None

    def getTokens(self, text):
        text = str(text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 3]

        return tokens

    def getTopics(self, text, num_topics, num_words):
        self.tokens = self.getTokens(text)
        self.dictionary = corpora.Dictionary([self.tokens])
        self.corpus = [self.dictionary.doc2bow(self.tokens)]

        self.model = models.LdaModel(corpus=self.corpus,
                                     num_topics=num_topics,
                                     id2word=self.dictionary)

        topics = [word for word, prob in self.model.show_topic(0, topn=num_words)]
        topics = " ".join(topics)

        return topics


class MainApp(App):
    def build(self):
        return Builder.load_file("main.kv")

    def clearPredictions(self, widget):
        widget.clear_widgets()

    def saveResults(self):
        path = filechooser.save_file()
        if path:
            pd.DataFrame(RESULTS).to_csv(f"{path[0]}.csv", index=False)


if __name__ == "__main__":
    MainApp().run()
