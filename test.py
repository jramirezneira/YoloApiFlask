from flask import Flask, jsonify, request, Response

app = Flask(__name__)
print("This line will be printed.")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True,  port=5001)