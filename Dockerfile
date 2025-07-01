# Utiliser l'image officielle Python comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source dans le container
COPY . /app/

# Définir le port utilisé par l'application
EXPOSE 5000

# Commande à exécuter lorsque le container démarre
CMD ["python", "app.py"]
