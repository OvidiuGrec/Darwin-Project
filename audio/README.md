# Using openSMILE

To run openSMILE:

execute 

$ Darwin-Project/openSMILE/msvcbuild/SMILExtract_Release.exe -C Darwin-Project/openSMILE/opensmile/config/avec2013.conf -I <sound file> -O Darwin-Project/audio/results.csv

config files need to be of type .conf and many are provided in Darwin-Project/openSMILE/opensmile/config (AVEC 2013 is included in that folder)

sound files can be anything .wav and maybe other types as well (haven't tried)

output files is the .csv you want to write over or create

you may or may not need to use absolute paths for your files, depending on your shell