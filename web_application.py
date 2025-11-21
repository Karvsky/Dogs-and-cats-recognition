import os
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from main import DL_MODEL

app = Flask(__name__)
app.secret_key = 'tajny_klucz_aplikacji'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = DL_MODEL()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    wynik = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nie przesłano pliku', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('Nie wybrano pliku', 'warning')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            sciezka_bezwzgledna = os.path.join(os.path.abspath(UPLOAD_FOLDER), filename)

            file.save(sciezka_bezwzgledna)

            wynik = model.image_decision(sciezka_bezwzgledna)

            flash('Plik został przetworzony pomyślnie!', 'success')
        else:
            flash('Niedozwolony format pliku', 'danger')

    return render_template('index.html', result=wynik)


if __name__ == '__main__':
    app.run(debug=True)