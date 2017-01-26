import pandas as pd

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.user',
    sep='|', names=u_cols)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
    sep='\t', names=r_cols)

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.item',
    sep='|', names=m_cols, usecols=range(5))

ourschema = """
DROP TABLE IF EXISTS "users";
DROP TABLE IF EXISTS "ratings";
DROP TABLE IF EXISTS "movies";
CREATE TABLE "users" (
    "user_id" INTEGER PRIMARY KEY NOT NULL ,
    "age" INTEGER,
    "sex" VARCHAR,
    "occupation" VARCHAR,
    "zip_code" VARCHAR
);
CREATE TABLE "movies" (
    "movie_id" INTEGER PRIMARY KEY NOT NULL ,
    "title" VARCHAR,
    "release_date" VARCHAR,
    "video_release_date" FLOAT,
    "imdb_url" VARCHAR
);
CREATE TABLE "ratings" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "user_id" INTEGER NOT NULL,
    "movie_id" INTEGER NOT NULL,
    "rating" INTEGER,
    "unix_timestamp" INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(movie_id) REFERENCES movies(movie_id)
);
"""

from sqlite3 import dbapi2 as sq3
import os

PATHSTART = "."


def get_db(dbfile):
    sqlite_db = sq3.connect(os.path.join(PATHSTART, dbfile))
    return sqlite_db


def init_db(dbfile, schema):
    """Creates the database tables."""
    db = get_db(dbfile)
    db.text_factory = str
    db.cursor().executescript(schema)
    db.commit()
    return db


db = init_db("movielens.db", ourschema)
movies.to_sql("movies", db, if_exists="append", index=False)
ratings.to_sql("ratings", db, if_exists="append", index=False)
users.to_sql("users", db, if_exists="append", index=False)

# example
print movies.head()

# Check that all records in place
sel = "SELECT * FROM movies;"
c = db.cursor().execute(sel)
print len(c.fetchall())
print movies.shape

# We can add new data
ins = "INSERT INTO movies (movie_id, title, release_date, video_release_date, imdb_url) VALUES (?,?,?,?,?);"
valstoinsert = (-1, 'My super cool Data Science Movie', 'today!', None, 'nope, no url')
db.cursor().execute(ins, valstoinsert)
db.commit()
print db.cursor().execute("SELECT * FROM movies;").fetchall()[0]

# We can delete data we just added
rem = "DELETE FROM movies WHERE movie_id=-1;"
c = db.cursor().execute(rem)
db.commit()
print db.cursor().execute("SELECT * FROM movies;").fetchall()[0]


def make_query(sel):
    c = db.cursor().execute(sel)
    return c.fetchall()


print make_query("SELECT * FROM movies;")

# Different ways of selecting
print movies.head(5)

print movies.query("movie_id>1 & movie_id < 5")

print movies[(movies.movie_id > 1) & (movies.movie_id < 5)]

# Get info on db table
print make_query("PRAGMA table_info(movies);")
movie_cols = [e[1] for e in make_query("PRAGMA table_info(movies);")]
print movie_cols


# Helper method
def make_frame(list_of_tuples, legend=movie_cols):
    framelist = []
    for i, cname in enumerate(legend):
        framelist.append((cname, [e[i] for e in list_of_tuples]))
    return pd.DataFrame.from_items(framelist)


# query to DataFrame
out = make_query("SELECT * FROM movies WHERE movie_id > 1 AND movie_id < 5;")
print out
print make_frame(out).head()

# NaN and None and null etc same
out = make_query("SELECT * FROM movies WHERE video_release_date IS NULL;")
print make_frame(out).head()

# we can use LIKE clause
print movies[movies.title.str.contains('1996')].head()
out = make_query("SELECT * FROM movies WHERE title LIKE '%1996%';")
print make_frame(out).head()

# We can select list of values
out = make_query("SELECT * FROM movies WHERE release_date IN ('01-Jan-1995','01-Jan-1996');")
print make_frame(out).head()

print movies[movies.release_date.isin(['01-Jan-1995', '01-Jan-1996'])].head()

# between clause
out = make_query("SELECT * FROM movies WHERE movie_id BETWEEN 10 AND 50;")
print make_frame(out).head()

# query
print movies.query("10 <= movie_id <= 50").head()

# sorting
print movies.sort_values('title').head()
print make_frame(make_query("SELECT * FROM movies ORDER BY title;")).head()
print movies.sort_values('title', ascending=False).head()
print make_frame(make_query("SELECT * FROM movies ORDER BY title DESC;")).head()

# select columns
print movies[['title', 'release_date']].head()
print make_frame(make_query("SELECT title, release_date FROM movies;"),
                 legend=['title', 'release_date']).head()

# unique selection
print movies['release_date'].drop_duplicates().count()
print movies['release_date'].drop_duplicates().head()
print make_frame(make_query("SELECT DISTINCT release_date FROM movies;"),
                 legend=['release_date'])['release_date'].head()

# add new columns
print users.head()
users['sex_occupation'] = users['sex'] + ':' + users['occupation']
users.assign(sex_occupation2=users['sex'] + ':' + users['occupation'])
print users.head()
# humm...
print users.assign(sex_occupation2=users['sex'] + ':' + users['occupation']).head()

# users.loc[] vs users[]

alt = "ALTER TABLE users ADD COLUMN sex_occupation;"
db.cursor().execute(alt)
print make_query("PRAGMA table_info(contributors);")

out = make_query("SELECT user_id, sex, occupation FROM users;")
out2 = [(e[1] + ":" + e[2], e[0]) for e in out]
print out2

alt2 = "UPDATE users SET sex_occupation = ? WHERE user_id = ?;"
for ele in out2:
    db.cursor().execute(alt2, ele)
db.commit()
print make_frame(make_query("SELECT * FROM users;"), u_cols + ["sex_occupation"]).head()

# aggregation
print ratings.describe()
print ratings.rating.max()

out = make_query("SELECT MAX(rating) AS maxrating FROM ratings;")
print make_frame(out, ['maxrating'])

print ratings.groupby("movie_id")['movie_id', 'rating'].mean()
out = make_query("SELECT movie_id, AVG(rating) AS avgr FROM ratings GROUP BY movie_id;")
print make_frame(out, legend=['movie_id', 'avgr'])

# Remove rows
movies2 = movies.copy()
print movies2.head()
movies2.set_index('title', inplace=True)
movies2.drop(['Toy Story (1995)'], inplace=True)
print movies2.head()
movies2.reset_index(inplace=True)
print movies2.head()

# Remove rows (better)
print movies[movies.title != 'Toy Story (1995)'].head()

# limits
print make_frame(make_query("SELECT * FROM movies LIMIT 3"))
print movies[0:3]
print movies.head(3)

# joins
# inner
out = make_query(
    "SELECT ratings.movie_id, ratings.rating, movies.title FROM ratings, movies WHERE ratings.movie_id = movies.movie_id")
print make_frame(out, legend=['ratings.movie_id', 'ratings.rating', 'movies.title'])
print ratings.merge(movies, on='movie_id')

# left
query_join_str = "SELECT ratings.movie_id, ratings.rating, movies.title FROM ratings LEFT JOIN movies ON ratings.movie_id = movies.movie_id"
out = make_query(query_join_str)
print make_frame(out, legend=['ratings.movie_id', 'ratings.rating', 'movies.title'])
print ratings.merge(movies, on='movie_id', how='left')

print pd.read_sql(query_join_str, db).head()

db.close()
