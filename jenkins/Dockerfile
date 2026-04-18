# =============================================================================
# Jenkins Dockerfile — SITARAM Platform
# Base: Jenkins LTS + Python + PostgreSQL client + Redis CLI + Kafka tools
# =============================================================================

FROM jenkins/jenkins:lts-jdk17

USER root

# Install system dependencies Jenkins needs for validation scripts
RUN apt-get update && apt-get install -y \
    python3 python3-pip \          
    postgresql-client \            
    redis-tools \                  
    curl wget jq \                 
    kafkacat \                     
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for metric fetching scripts
RUN pip3 install psycopg2-binary redis anthropic pandas numpy --break-system-packages

# Install Jenkins plugins — all plugins needed for SITARAM pipeline
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt

# Pre-configure Jenkins (skip setup wizard — production mode)
ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false"

# Copy pipeline definition
COPY pipelines/ /var/jenkins_home/pipelines/

# Copy Jenkins job configuration
COPY jobs/ /usr/share/jenkins/ref/jobs/

USER jenkins

EXPOSE 8080 50000
