from flask import Flask, render_template, request
import configparser

app = Flask(__name__)

CONFIG_FILE = 'aiblue.conf'

def load_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def save_config(config):
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/config', methods=['GET', 'POST'])
def config():
    config = load_config()
    if request.method == 'POST':
        config.set('Configuration', 'persona', request.form['persona'])
        config.set('Configuration', 'name', request.form['name'])
        config.set('Configuration', 'creator', request.form['creator'])
        config.set('Configuration', 'style', request.form['style'])
        config.set('Configuration', 'top_priority', request.form['top_priority'])
        config.set('History', 'max_length', request.form['max_length'])
        config.set('History', 'store', request.form['store'])
        config.set('History', 'recall', request.form['recall'])
        config.set('Instructions', 'instructions', request.form['instructions'])
        config.set('AI_Configuration', 'main_ai', request.form['main_ai'])
        config.set('AI_Configuration', 'subunit1', request.form['subunit1'])
        config.set('AI_Configuration', 'subunit2', request.form['subunit2'])
        save_config(config)

        return redirect(url_for('home'))
    return render_template('index.html', config=config)

@app.route('/run')
def run():
    # Code to run AIBlue
    return "Running AIBlue"

if __name__ == '__main__':
    app.run(port=7860)
