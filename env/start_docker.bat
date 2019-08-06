@echo off

set dtbpath=C:\Program Files\Docker Toolbox\
set gitpath=C:\Program Files\Git\

cd /d "%dtbpath%"

SET tmp=%~dp0

SET mypath=%tmp:~1,-5%
set mypath=%mypath::=%
set mypath=%mypath:\=/%

set disk=%tmp:~0,1%

for %%a in ("A=a" "B=b" "C=c" "D=d" "E=e" "F=f" "G=g" "H=h" "I=i"
            "J=j" "K=k" "L=l" "M=m" "N=n" "O=o" "P=p" "Q=q" "R=r"
            "S=s" "T=t" "U=u" "V=v" "W=w" "X=x" "Y=y" "Z=z" "Ä=ä"
            "Ö=ö" "Ü=ü") do (
    call set disk=%%disk:%%~a%%
)

"%gitpath%bin\bash.exe" --login -i "%dtbpath%start.sh"^
	docker run^
		-it^
		--rm^
		-p 8888:8888^
		-v /%disk%%mypath%:/home/jovyan/jupyter^
		cavsim^
		/app/startup.sh
		