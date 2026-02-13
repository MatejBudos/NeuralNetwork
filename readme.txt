Začal som s rozdelením dát na validačnu a trénovaciu množinu tak, že som zoznam zamiešal a potom rozdelil
v pomere 80:20.
Pokračoval som vytvorením rôznych aktivačných funkcíí nech ich mám pripravené.
Na youtube som si pozral rôzne návody ako by mal celý ten proces vytvárania 
siete fungovať. Spravil som v class layer základnú inicializáciu
a compute output, kde som väčšinu vecí prebral z úlohy o perceptrone. Inicialízácia váh sa volá 
Xavier weight initialization a čo som sa dočítal tak by mala byť efektívnejšia spolu so sigmoid 
alebo tanh aktivačnými funkciami no nejaký výrazný rozdiel som nepostrehol.
Štyl prepojenia siete a jej prechádzanie som prebral z jedného návodu a prispôsobil som si ho podľa seba.
Backwards metoda mi robila najväčší problém implementovať čiže je to kombinacia 
aplikovania vzorcov zo zadania a opravy od chatu gpt.
Potom som zacal s train metodou, kde som princip prebral z perceptron zadania a neskôr som tam
dorobil ukladanie najlepšieho výsledku. Earlystopping mi moc nedával zmysel kedže mi veľa krát 
MSE modelu osciluje okolo jednej hodnoty aj niekoľko desiatok epóch a potom sa error zníži.
Čo sa týka rôznych aktivačných funkcií tak rozdiel medzi tanh a sigmoid tak so sigmoid sa mi podarilo
natrénovať lepšiu sieť, ale nejaký výrazný rozdiel som v priebehu učenia som nevidel.
Myslím si že hlavnú úlohu tu zohravala veľkosť siete a počet neurónov vo vrstve,
Najviac sa mi osvedčili 4 vrstvy po 32 neuronov v tomto formate:
2 -> 32 : Sigmoid
32 -> 32 : Sigmoid
32 -> 32 : Sigmoid
32 -> 1 : Linear

Moc som modely netrénoval na epochy ale nas čas 5 minut a loop sa potom breakol.
Zvyčajne išlo o cca 1200 epoch. 
Na menej vrstvových aj 2300epoch ale nedosiahli takú úspešnosť ako 4 vrstvové

Na koniec som dorobil ukladanie a načítavanie váh a biases do modelu, spolu s testovacou funkciou.



