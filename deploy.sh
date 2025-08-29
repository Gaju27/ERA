#!/bin/bash
set -e

# Update system
sudo yum update -y
sudo yum install -y python3 python3-pip wget unzip

# Setup app directory
mkdir -p ~/flaskapp
cd ~/flaskapp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install flask gunicorn

# Create static folder for images
mkdir -p static

# Download stable animal images from Wikimedia Commons
wget -O static/cat.jpg https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
wget -O static/dog.jpg https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg
wget -O static/elephant.jpg https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg

# Write Flask app
cat > app.py << 'EOF'
from flask import Flask, request, jsonify, send_from_directory, Response

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Animal Selector + File Upload</title>
</head>
<body>
  <h1>Animal Selector + File Upload</h1>
  
  <div>
    <h3>Select animal:</h3>
    <form id="animalForm">
      <input type="checkbox" name="animal" value="cat"> Cat<br>
      <input type="checkbox" name="animal" value="dog"> Dog<br>
      <input type="checkbox" name="animal" value="elephant"> Elephant<br>
    </form>
    <div id="preview"><em>Pick an animal</em></div>
  </div>

  <hr>
  
  <div>
    <h3>Upload file:</h3>
    <input id="fileInput" type="file"/>
    <button onclick="uploadFile()">Upload</button>
    <pre id="result"></pre>
  </div>

<script>
const form = document.getElementById('animalForm');
form.addEventListener('change', (e) => {
  [...form.elements].forEach(el => { if (el !== e.target) el.checked = false; });
  if(e.target.checked){
    const animal = e.target.value;
    const imgUrl = '/image/' + animal;
    document.getElementById('preview').innerHTML = '<img src="'+imgUrl+'" style="max-width:400px; height:auto;">';
  } else {
    document.getElementById('preview').innerHTML = '<em>Pick an animal</em>';
  }
});

async function uploadFile(){
  const f=document.getElementById('fileInput').files[0];
  if(!f){alert("Choose file");return;}
  const fd=new FormData(); fd.append("file",f);
  let res=await fetch('/upload',{method:'POST',body:fd});
  let j=await res.json();
  document.getElementById('result').textContent=JSON.stringify(j,null,2);
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return INDEX_HTML

@app.route('/image/<animal>')
def image(animal):
    if animal not in ["cat","dog","elephant"]:
        return Response("Not found", status=404)
    return send_from_directory("static", f"{animal}.jpg")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return jsonify({"error":"No file"}),400
    f=request.files['file']
    data=f.read()
    return jsonify({"name":f.filename,"size":len(data),"type":f.mimetype})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
EOF

# Kill old gunicorn if running
pkill gunicorn || true

# Run with gunicorn on port 80
sudo venv/bin/gunicorn -b 0.0.0.0:80 app:app --daemon

echo "✅ Deployment complete!"
echo "Open: http://<your-ec2-public-ip>/"
