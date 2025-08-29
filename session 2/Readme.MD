# Flask File Upload and Animal Selector App

This is a simple Flask web application with a **frontend (HTML/CSS/JS)** and a **Flask backend**.  

Features:
- Select between **Cat, Dog, or Elephant** → shows the corresponding image.  
- Upload any file → shows its **file name, size, and type**.  

---

## 🚀 Deploy on AWS EC2

### 1. Launch EC2 Instance
- Use **Amazon Linux 2** or **Ubuntu 22.04**  
- Instance type: `t2.micro`  
- Configure security group to allow:  
  - **Port 22** (SSH)  
  - **Port 5000** (Flask) or **Port 80** 

---

### 2. Connect to EC2

**Amazon Linux**:
```bash
ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP>

nano deploy.sh
chmod +x deploy.sh
./deploy.sh
````
#### 3. Folder structure
```flask-app/
│-- app.py
│-- static/
│    └── images/
│         ├── cat.jpg
│         ├── dog.jpg
│         └── elephant.jpg
│-- templates/
     └── index.html
