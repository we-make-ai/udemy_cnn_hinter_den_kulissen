# Kursrepository zu Udemy Kurs "Convolutional Neural Networks - Hinter den Kulissen"

Dieses Repository beinhaltet die Jupyter Notebooks zu dem online Kurs **"[Convolutional Neural Networks - Hinter den Kulissen](https://academy.we-make.ai/courses/cnn-hinter-den-kulissen/)"**.

Wir erarbeiten in diesem Online Training zu künstlicher Intelligenz, wie CNNs (Convolutional Neural Networks) aufgebaut sind und betrachten die verschiedenen erforderlichen Layer wie _Convolution_, _Batch Normalization_ und _Pooling Layer_.

Zusätzlich erarbeiten wir eine zusätzliche Variante von CNNs: **Residual Networks**. Wir Analysieren den Aufbau von Residual Networks und implementieren selbst die erforderlichen Layer mithilfe von reinem Python Code. Natürlich vergleichen wir - wie in jedem unserer Online-Kurse zu künstlicher Intelligenz - unsere eigene Implementierung mit jener aus den Frameworks _PyTorch_ und _fastai_.

## Inhalte dieses Online-Kurses

* Aufbau von Convolutional Neural Networks in der Theorie und Praxis
* Die Convolution: Wir programmieren einen Convolutional Pooling Layer (Conv2d)
* Berechnung von Paddings und Strides bei Convolutions
* Die mathematischen Gleichungen hinter Convolutions
* Wie ist der CNN-Learner aus dem fastai Framework aufgebaut?
* Welche Theorien und Funktionalitäten sind in der 1cycleTraining Methode vom fastai Framework enthalten?
* Warum hilft das Konzept der Batch Normalization einem Convolutional Neural Network, das Training zu beschleunigen?
* Was ist eine Residual Network Architektur (ResNet) und wofür kann ich diese einsetzen?
* Wir implementieren unser eigenes Resdiual Net Model sowohl mithilfe von PyTorch wie auch mit reinem Python Code.
* Skip Connections ist ein Konzept, das Residual Networks optimieren kann. Wir analysieren, was Skip Connections darstellen und wie wir diese in unserem Modell implementieren können.

## Hintergrund zu Convolutional Neural Networks - ConvNets

Warum sollte ich mich im Zuge eines Kurses über künstliche Intelligenz auch mit dem Thema Convolutional Neural Net (ConvNets) beschäftigen?

Insbesondere im Bereich der Analyse und Verarbeitung von Bilddaten konnte künstliche Intelligenz in den vergangenen Jahren enorme Erfolge verbuchen. Besonders gestützt wurde diese Entwicklung durch die Entwicklungen im Bereich neuronaler Netze hin zu Convolutional Neural Nets. CNNs trugen massiv dazu bei, dass die Lücke in der Bildverarbeitung (Computer Vision) zwischen den kognitiven Fähigkeiten von Mensch und Maschine zunehmend verschwindet.

Mithilfe von ConvNets ist die künstliche Intelligenz einen Schritt weiter, Aufgaben aus dem Bereich Computer Vision ähnlich wie menschliche Anwendung erledigen und umsetzen zu können.

Das Ziel für künstliche Intelligenz Algorithmen ist es, dass Maschinen die Welt genauso sehen können wie wir Menschen auch und dieses Wissen für eine Vielzahl von Aufgaben wie Bilderkennung, Bildanalyse, Bildklassifizierung, etc. nützen können. Die dafür schlagende technologische Entwicklung im Bereich von deep learning stellen neuronale Netze - und zwar Convolutional Neural Nets - dar.

Ein Convolutional Neural Net ist ein deep Learning Algorithmus der mit Bilddaten arbeitet. Der Algorithmus lernt, die Gewichte und Bias-Werte im neuronalen Netz entsprechend der Wichtigkeit der einzelnen Features anzupassen. Das Vorverarbeiten der Bilddaten ist im Vergleich zu anderen Klassifizierungsalgorithmen durch den Einsatz des neuronalen Netzes weniger wichtig, um gute Erfolge erzielen zu können. Während in einfachen Algorithmen die anzuwendenden Filter manuell implementiert werden, lernt das Convolutional Neural Net diese Filter bzw. Eigenschaften selbst zu entwickeln.

Die grundsätzliche Architektur eines neuronalen Netzes ist von der Funktionsweise und Vernetzung der Synapsen im menschlichen Gehirn inspiriert. Die ConvNet (Convolutional Neural Net) Architektur wiederum fand ihren Ursprung in der Organisation des visuellen Kortex. Einzelne Neuronen reagieren nur auf Stimuli in einem stark begrenzten Bereich des visuellen Umfelds - bezeichnet als rezeptives Feld (receptive field). Um das gesamte Sehfeld abzudecken, existiert eine Vielzahl solcher Bereiche, die sich gegenseitig überlappen.

Ein Bild wird auch in ein Convolutional Neural Net als Pixelmatrix eingeführt. Ein Ansatz wäre natürlich, diese Matrix in einen eindimensionalen Vektor zu transponieren und durch ein Multi-Level-Perzeptron Netz zur Klassifikation zu schleusen. Das könnte für sehr einfache Bilder zwar durchaus funktioniert, repräsentiert aber weder die Erfolge der letzten Jahre im Deep Learning Computer Vision Bereich, noch würde es für komplexere Bilddaten ausreichend gut funktionieren.

Ein deep Convolutional Neural Net (neuronales Netz) stellt eine Möglichkeit dar, sowohl zeitliche, wie auch räumliche Abhängigkeiten mithilfe von Filtern in Bildern darzustellen. Ein ConvNet ist in der Lage, die Anzahl der Parameter im Modell zu reduzieren und die Gewichtung über die Trainingsvorgänge hinweg anzupassen. So hat ein convolutional neural net die Aufgabe, die Dimension von Bilddaten ohne Inhaltsverlust zu reduzieren. Auf diese Weise erhalten wir ein Modell, das nicht nur mit einer hohen Genauigkeit Bilder klassifizieren kann, sondern auch sehr gut skalierbar ist, bei einer hohen Datenmenge.

Die wichtigsten Varianten bzw. konkreten Ausprägungen von convolutional neural network Architekturmodellen sind: LeNet, AlexNet, VGGNet (VGG16, und andere), GooLeNet, ResNet, ZFNet.

Im zweiten Teil des Kurses widmen wir uns dem Thema Residual Nets (ResNets). Es war im Jahr 2012, dass Krizhevsky et al. den Weg für deep convolutional networks frei machten. Die Tiefe des neuronalen Netzes betrug damals ganze acht Layer - davon fünf Convolutions und drei lineare Layer (fully connected layer). In dem Paper zeigten die Autoren, dass allein mit der Erhöhung der Anzahl an Layer ab einer gewissen Gesamtanzahl keine nennenswerten Verbesserungen im Trainingsergebnis des tiefen neuronalen Netzes mehr erreichbar sind. Die Evidenz zeigt, dass die optimale Gesamtanzahl von Layern für Modellen, die auf dem ImageNet Modell aufbauen zwischen 16 und 30 liegt.

Die Schwierigkeit, tiefe neuronale Netze zu trainieren wurde durch das Einführen des Konzepts eines residual Blocks entschärft. Der residual Block ist mithilfe eines Konzepts mit der Bezeichnung "Skip Connection" in der Lage, die Ausgabewerte eines Layers mit jenen des nächsten Layers zu verknüpfen und bestimmte "Kurzschlüsse" innerhalb des convolutional Networks zu schaffen. Die Autoren des entsprechenden Papers, in dem residual blocks mit skip connections vorgestellt wurden, zeigten, dass sie so in der Lage waren, neuronale Netze mit 100 bis 1000 Layer erfolgreich zu trainieren. Auch der Test auf dem ImangeNet Datenset zeigte, dass das ResNet Modell mit 152 Layern weniger trainierbare Parameter als das VGG Modell aufweist.

Skip Connections stellen also eine sehr interessante Erweiterung von deep convolutional networks dar. Die Autoren haben den Erfolg der ResNet Modellarchitektur hinsichtlich der Klassifizierung von Bilddaten ausdrücklich dargelegt.
