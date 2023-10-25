from flask import Flask, render_template_string

app = Flask(__name__)

# Sample DataFrame as a dictionary for simplicity
df = {
    1: {'title': 'Movie A', 'info': 'Genre: Action', 'photo': 'photo_A_url'},
    2: {'title': 'Movie B', 'info': 'Genre: Drama', 'photo': 'photo_B_url'},
    3: {'title': 'Movie C', 'info': 'Genre: Romance', 'photo': 'photo_C_url'}
}
@app.route('/movies')
def movies():
    return render_template_string(movies_template, movies=df.values())

movies_template = """
<html>
    <head>
        <style>
            .movie {
                position: relative;
                display: inline-block;
                padding: 10px;
                border: 1px solid #ccc;
                margin: 5px;
                width: 150px;
                text-align: center;
            }

            .movie .info-panel {
                visibility: hidden;
                background-color: #555;
                color: #fff;
                text-align: center;
                padding: 5px;
                position: absolute;
                z-index: 1;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
            }

            .movie:hover .info-panel {
                visibility: visible;
                opacity: 1;
            }
        </style>
    </head>
    <body>
        {% for movie in movies %}
            <div class="movie">
                <span>{{ movie.title }}</span>
                <div class="info-panel">
                    {{ movie.info }}<br>
                    <img src="{{ movie.photo }}" width="100">
                </div>
            </div>
        {% endfor %}
    </body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)

