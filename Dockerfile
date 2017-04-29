FROM python

MAINTAINER Sneha Ravikumar <ravikumar.s@husky.neu.edu>
WORKDIR /src

RUN apt-get update && \
    apt-get clean && \
            rm -rf /var/lib/apt/lists/*

USER root
	        
		# Install Python 3 packages
		# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
		# use notebook-friendly backends in these images
RUN pip install 'matplotlib' \
		'numpy' \
		'keras' \
		'h5py' 
RUN apt-get update \
	&& apt-get install -y wget \ 
	'mercurial' \
	'libfreetype6-dev' \
	'libsdl1.2-dev' \
	'libsdl-image1.2-dev' \ 
	'libsdl-ttf2.0-dev' \
	'libsmpeg-dev' \
	'libportmidi-dev' \
	'libavformat-dev' \
	'libsdl-mixer1.2-dev' \
	'libswscale-dev' \
	'libjpeg-dev' \
	'python-pygame' 	

									
WORKDIR /src
COPY . /src
EXPOSE 8123		
CMD ["bash"]
