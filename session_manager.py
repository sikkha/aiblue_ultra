# session_manager.py
from flask import Flask, request, jsonify, make_response
import sqlite3
import uuid

app = Flask(__name__)

def init_db():
    # Initialize the database and create tables if they don't exist
    conn = sqlite3.connect('sessions.db')
    cursor = conn.cursor()
    
    # Create 'user_sessions' table with the new design
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        interaction TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

app = Flask(__name__)

@app.route('/start_session', methods=['GET'])
def start_session():
    # Generate a unique session_id
    session_id = str(uuid.uuid4())

    # Connect to the SQLite database
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()

    # Note: No need to insert a record here since this is just starting the session.
    # The 'user_sessions' table will be updated when interactions occur.

    conn.close()  # Close the database connection

    # Create a response indicating that the session has started and include the session_id
    resp = make_response(jsonify({"message": "Session started", "session_id": session_id}))

    # Set a cookie in the response with the session_id to identify the session
    resp.set_cookie('SessionID', session_id)

    # Return the response to the client
    return resp


@app.route('/get_session', methods=['GET'])
def get_session():
    # Retrieve the session_id from the cookies
    session_id = request.cookies.get('SessionID')
    
    # Connect to the SQLite database
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    
    # Fetch the history for the given session_id from the user_sessions table
    c.execute("SELECT history FROM user_sessions WHERE session_id = ?", (session_id,))
    history = c.fetchone()
    
    # Close the database connection
    conn.close()
    
    # Check if history was found for the session_id
    if history:
        # Return the history as JSON
        return jsonify({"history": history[0]})
    else:
        # If no history is found, return an error message
        return jsonify({"error": "Session not found"})


@app.route('/get_latest_sessions', methods=['GET'])
def get_latest_sessions():
    # Retrieve the session_id from the cookies
    session_id = request.cookies.get('SessionID')
    number_of_entries = request.args.get('number', default=5, type=int)  # Default to the latest 5 entries if not specified

    # Connect to the SQLite database
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()

    # Fetch the latest 'number_of_entries' interactions for the given session_id
    c.execute('SELECT interaction, timestamp FROM user_sessions WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?', (session_id, number_of_entries))
    interactions = c.fetchall()

    # Close the database connection
    conn.close()

    # Prepare the interactions for JSON serialization
    formatted_interactions = [{'interaction': interaction[0], 'timestamp': interaction[1]} for interaction in interactions]

    # Check if any interactions were found
    if interactions:
        # Return the latest interactions as JSON
        return jsonify({"interactions": formatted_interactions})
    else:
        # If no interactions are found, return an error message
        return jsonify({"error": "No session interactions found"})


@app.route('/update_session', methods=['POST'])
def update_session():
    # Retrieve the session_id from the cookies
    session_id = request.cookies.get('SessionID')
    new_interaction = request.json.get('new_interaction', '')

    # Connect to the SQLite database
    conn = sqlite3.connect('sessions.db')
    cursor = conn.cursor()

    # Insert a new interaction into the user_sessions table for the given session_id
    cursor.execute('INSERT INTO user_sessions (session_id, interaction) VALUES (?, ?)', (session_id, new_interaction))

    # Commit the changes to the database and close the connection
    conn.commit()
    conn.close()

    # Return a message indicating the interaction was added to the session
    return jsonify({"message": "Interaction added to session"})

if __name__ == '__main__':
    init_db()  # Ensure the database and tables are initialized
    #app.run(port=5001)
    app.run(port=5001, debug=True)
