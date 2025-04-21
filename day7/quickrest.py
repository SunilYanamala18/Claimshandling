from flask import Flask, request, render_template
import os
app=Flask(__name__)
f_path=r".\content.txt"
@app.route("/")
def index():
    return """
    <html>
        <body>
            <h1>Welcome to the home page :)</h1>
        </body>
    </html>
    """
@app.route("/updatefortoday", methods=['GET','POST'])
def update():
    if request.method == 'POST':
        con = request.form.get('content',None)
        print(con)
        
        with open(f_path,'a') as f:
            f.write(con)
        return """
            <html>
                <body>
                    <h1>Your form submitted successfulyy....:)</h1>
                </body>
            </html>
        """
    return """
    <html>
        <body>
            <h1>Write your content</h1>
            <form method='post' action='/updatefortoday'>
                <textarea cols=150 rows=30 name='content'> </textarea></br></br>
                <input type='submit' value='Submit'>
            </form>
        </body>
    </html>
    """
@app.route("/share", methods=['GET'])
def share():
    with open(f_path,'r') as f:
        lines = f.read()
    # d = dict(content = lines)
    print(lines)
    return render_template("shared.html",content=lines)
@app.route("/clearnotepadtxt", methods=['GET'])
def clear():
    with open(f_path,'w') as f:
        f.write('')
    return "Content cleared successfully :)"
if __name__ == '__main__':
    app.run()