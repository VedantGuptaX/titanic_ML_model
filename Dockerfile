FROM python:3-slim

#set working directory
WORKDIR /app
#Copy contents
COPY . .
#Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt
#Expose Ports
EXPOSE 8000
#Start API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
