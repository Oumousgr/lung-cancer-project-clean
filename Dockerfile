# Exemple de Dockerfile pour configurer AWS dans Docker
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Passer les variables d'environnement AWS
ENV AWS_ACCESS_KEY_ID=your-access-key-id
ENV AWS_SECRET_ACCESS_KEY=your-secret-access-key
ENV AWS_DEFAULT_REGION=us-east-1  # Remplace par ta région AWS

# Copier le code source dans le container
COPY . /app/

# Exposer le port
EXPOSE 5000

# Commande à exécuter lorsque le container démarre
CMD ["python", "app.py"]
