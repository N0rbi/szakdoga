# Bevezetés

A rekurrens neurális hálók már a 80-as években megjelentek. Ezek olyan neurális hálók,
 melyek figyelembe veszik az előző állapokokat a döntéshozatalban. A ma használt rekurrens hálók közül az lstm azaz a 
long shot-term memory a legkedveltebb mind közül, mert megolást talál az rnn-ek egy alapvető problémájára,
a gradiensek drasztikus növekedésére vagy csökkenésére, melyek ellehetetlenitik a hosszútávú tanulást. Ezt a hálótipust alkalmazom dolgozatomban.

Dolgozatom célja lstm rétegekkel létrehozni egy model-t, amely képes megtanulni egy adott előadó zenei stilusát karakterek sorozatát nézve.
Ponotsabban felteszi magában a kérdést: "ha ezt az x hosszú szöveget látom, vajon az előadó mit irna x+1. karakternek?".
A model létrehozásában a python nyelven elérhető keras és annak hátterében a tensorflow keretrendszereket használom.
Keras egy API amely elfedi a neurális hálókhoz szükséges matematikát, igy átláthatóbbá téve a kódot, tensorflow pedig 
egy eszköz mellyel gépi tanuló szoftvereket könnyedén tudsz tanitani gyorsasága miatt, valamint
átláthatóvá teszi a fejlesztést a tensorboard segitségével, mely egy vizualizációs eszköz.

Szakdolgozatomban először ismertetem az egyszerű neurális hálókat, működésüket, 
majd ismertetem a rekurrens hálókat azok hasznát, és kitérek a problémájukra melyet az lstm old meg.
Ezután ismertetem a keras keretrendszerét, a tensorflow működését és ezen belül a tensorboard-ot.
Ezek ismeretében már olvasható a tensorboard vizualizációja, igy megmutatom a tanitások eredményeit.