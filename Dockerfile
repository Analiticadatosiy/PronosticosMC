
FROM mcr.microsoft.com/dotnet/framework/runtime:4.8




ADD https://aka.ms/vs/17/release/vc_redist.x64.exe  /app/VC_redist.x64.exe
WORKDIR /app
RUN VC_redist.x64.exe /quiet /install
#RUN C:\VC_redist.x64.exe -Wait /install /passive /norestart

FROM winamd64/python:3.9-windowsservercore-ltsc2016


#USER ContainerAdministrator
#RUN curl -fSLo vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe \
#    start /w vc_redist.x64.exe /install /quiet /norestart \
#    del vc_redist.x64.exe

WORKDIR /app



COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
EXPOSE 1434
EXPOSE 1433
EXPOSE 62586

ENTRYPOINT ["streamlit", "run"]

CMD ["app_pronosticos.py"]