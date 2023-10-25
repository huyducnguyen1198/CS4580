from flask import Flask, request, render_template_string
import pandas as pd
from main import *
import requests
app = Flask(__name__)

@app.route('/handle_click')
def handle_click():
    imdbId = request.args.get('imdbId')
    imdbId = str(int(imdbId[2:]))
    l = loadPastMovies()
    l.append(imdbId)
    savePastMovies(l)
    return f'You clicked on {imdbId}'
@app.route('/', methods=['GET', 'POST'])
def index():
    ###### search using title ######

        ############check for tags##########
    # get data from csv and preprocess using methods in main.py
    tags = allGen
    df = pd.read_csv('movies.csv')
    df = extractYear(df)
    df = editIMDB(df)

    selected_tags = []
    selected_year = '0'
    if request.method == 'POST':

        selected_tags = request.form.getlist('tag_name')
        if 'year' in request.form:
            selected_year = request.form['year']

        print(selected_tags)
        print(selected_year)

    # Generate HTML content
    tag_checkboxes = ''.join([f'<span class="tag"><input type="checkbox" name="tag_name" '
                              f'value="{tag}">{tag}</span>' for tag in tags])
    selected_tags_display = f'<p>Movies By Genres: {", ".join(selected_tags)}</p>' if selected_tags else ''

        #############check for tag end ##########

        #### year start #####
    unique_years = sorted(df['year'].unique())
    year_options = ''.join([f'<option value="{year}">{year}</option>' for year in unique_years])
    selected_year_display = f'<p>Movies By year: {", ".join(selected_year)}</p>' if selected_year != '0' else ''

        #### year end #####


    table = getMovieRec(df, n=50)
    title = ""
    #search title
    if request.method == 'POST':
        title = request.form['title']
        if title:
            if selected_tags:
                table = genresimilarity(df, genres=selected_tags)[:50]
                table = titlesimilarity(title,df_t= table, n = 50)
            else:
                table = titlesimilarity(title, df_t = df, n = 50)
        else:
            if selected_tags:
                table = genresimilarity(df, genres=selected_tags)[:50]
            else:
                table = getMovieRec(df, n=50)
    if selected_year != 0:
        table = df[df['year'] == int(selected_year)]



    '''Try to extract imdb and title from df, and render it so that html print out box-like table'''
    mov_list = [{'imdbId': row['imdbId'],
                 'title': row['title'],
                 'genres': row['genres']}
                for index, row in table.iterrows()]
    #return render_template_string(template, table=table.to_html(index = False),
    # tag_checkboxes=tag_checkboxes, selected_tags_display=selected_tags_display)
    return render_template_string(movies_template, movies=mov_list,
                                  tag_checkboxes=tag_checkboxes,
                                  selected_tags_display=selected_tags_display,
                                  year_options=year_options,
                                  selected_year_display=selected_year_display)

'''async function handleMovieClick(imdbId) {
            fetch(`/handle_click?imdbId=${imdbId}`)
            .then(response => response.text())
            .then(data => {
                alert(data);  // Here we're just alerting the server's response for demonstration
            });}'''

movies_template = """
<html>
<head>
    <script>
        let currentPage = 1;
        const ITEMS_PER_PAGE = 10;
        movie = {{ movies|tojson }}
        console.log(movie);
        async function handleMovieClick(imdbId) {
            fetch(`/handle_click?imdbId=${imdbId}`)
            
            try {
                let response = await fetch(`https://www.omdbapi.com/?i=${imdbId}&apikey=4daa1e35`);
                let movieData = await response.json();
                console.log(movieData);

                // Assuming the API returns an object with a `posterLink` property
                window.open(movieData['Poster'], '_blank');
            } catch (error) {
                console.error("Failed to fetch movie data:", error);
            }
        }

        async function displayMovies() {
            let movies = {{ movies|tojson }};
            let start = (currentPage - 1) * ITEMS_PER_PAGE;
            let end = start + ITEMS_PER_PAGE;
            let displayedMovies = movies.slice(start, end);

            //get rating from api
            
            let moviesDiv = document.getElementById('moviesDiv');
            moviesDiv.innerHTML = '';
            for (let movie of displayedMovies) {
                //get rating from api

                try {
                    let response = await fetch(`https://www.omdbapi.com/?i=${movie.imdbId}&apikey=4daa1e35`);
                    let movieData = await response.json();
                    console.log(movieData);
                    //movie.rating = movieData['Ratings'][0].Value;
                    movie.rated = movieData['Rated'];
                    movie.img = movieData['Poster'];
                    // Assuming the API returns an object with a `posterLink` property
                    //window.open(movieData['Poster'], '_blank');
                } catch (error) {
                    console.error("Failed to fetch movie data:", error);
                }
            } 
            for (let movie of displayedMovies) {
                moviesDiv.innerHTML += `
                    <div class="movie">
                        <a href="#" onclick="handleMovieClick('${movie.imdbId}')">${movie.title} | ${movie.rated}</a>
                        <span class="info">${movie.genres}</span>
                    </div>`;    
            }
        }

        function nextPage() {
            currentPage++;
            displayMovies();
        }

        function prevPage() {
            currentPage--;
            displayMovies();
        }

        window.onload = function() {
            displayMovies();
        }
    </script>
    <style>
            body, html {
            height: 100%;
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container {
            width: 80%;
            max-width: 800px;
            padding: 20px;
            border: 1px solid #ddd;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            background-color: #f9f9f9;
        }
        .title-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
            border: 1px solid #ddd;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            background-color: #f9f9f9;
            margin-bottom: 20px;
        }
        .movie {
            position: relative;
            display: inline-block;
            height: 2em;
            padding: 5px;
            margin: 5px;
            border: 1px solid #ddd;
            margin: 8px; 
            font-size: 1em;
            text-align: center;
            line-height: 2em; /* This centers the text vertically */

            box-sizing: border-box; /* Ensuring padding doesn't affect the overall size of the box */
            padding: 0 10px; /* Adding some horizontal padding */

        }

        .info {
            visibility: hidden;
            background-color: #555;
            color: #fff;
            text-align: center;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            margin-bottom: 5px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .movie:hover .info {
            visibility: visible;
            opacity: 1;
        }
    </style>
    <style>
        .tag{
            display: inline-block;
            width: 20%;
        }
    </style>
</head>
<body>
    <div class="container">
        <form action="/" method="post">
            <label for="title">Search:</label>
            <input type="text" id="title" name="title">
            <label for="year">Year:</label>
            <select name="year" id="year">
                {{year_options|safe}}
            </select>
            <input type="submit" value="Search">
            <br>
            {{tag_checkboxes|safe}}
        </form>
        {{selected_tags_display|safe}}
        {{selected_year_display|safe}}

        <div class="title-container">
            <div id="moviesDiv"></div>
        </div>
        <button onclick="prevPage()">Previous</button>
        <button onclick="nextPage()">Next</button>
    </div>
</body>
</html>
"""


if __name__ == '__main__':
    app.run(debug=True)
    #getRequest("tt0111161")
