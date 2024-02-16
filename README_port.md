# Easy Curve Fit - App para Ajuste de Curvas

## Introdução

O ajuste de curvas é uma ferramenta essencial na engenharia química, permitindo que profissionais e pesquisadores modelagem processos e fenômenos complexos através de equações matemáticas. Este app foi desenvolvido com o objetivo de simplificar a aplicação de diferentes modelos de ajuste de curvas, como Linear, Exponencial, Modelo de Primeira Ordem, Função Logística Generalizada, Distribuição Granulométrica e Equação de Nagata, facilitando análises precisas e otimizadas em diversos contextos da engenharia química.

## Importância do Ajuste de Curvas na Engenharia Química

O ajuste de curvas é fundamental na engenharia química para a modelagem de processos, otimização de reações, controle de qualidade, e no desenvolvimento de novos materiais e produtos. A capacidade de prever comportamentos e entender profundamente as relações entre variáveis permite inovações e eficiências operacionais, destacando a importância de ferramentas como este app para profissionais da área.


## Exemplos de Modelos de Ajuste de Curvas

[![Watch the video](https://img.youtube.com/vi/qxRbjU4bems/0.jpg)](https://youtu.be/qxRbjU4bems)

### Modelo Linear

Ideal para relações diretas entre variáveis, oferecendo uma solução simples para análises iniciais de tendências.

### Modelo Exponencial

Aplicável em processos de crescimento ou decaimento que seguem uma taxa constante proporcional ao tamanho atual do sistema.

### Modelo de Primeira Ordem

Frequentemente usado em dinâmicas de sistemas onde a taxa de mudança é proporcional ao estado atual.

### Função Logística Generalizada

Utilizado para modelar crescimento populacional, processos de saturação e outros fenômenos que se aproximam de um limite máximo.

### Distribuição Granulométrica

Essencial para caracterizar a distribuição de tamanhos de partículas em misturas, comumente utilizado em análises de solo, sedimentos e materiais pulverizados.

### Equação de Nagata

Aplica-se à dinâmica de fluidos e ao estudo de fluxos, especialmente útil em engenharia química para modelar comportamentos de misturas e reações.

## Customização de Modelos

O diferencial deste app é a capacidade de criar e ajustar seus próprios modelos personalizados. Você não está limitado aos modelos predefinidos; nossa interface intuitiva permite que você defina qualquer tipo de equação para ajustar seus dados específicos, oferecendo flexibilidade sem precedentes em suas análises.

---

## Data Prep - Ramer-Douglas-Peucker

O algoritmo Ramer-Douglas-Peucker é um método usado em gráficos de computador e SIG (Sistema de Informações Geográficas) para simplificar curvas ou polilinhas reduzindo pontos. Foi desenvolvido independentemente por Urs Ramer em 1972 e por David Douglas e Thomas Peucker em 1973. O algoritmo funciona aproximando uma curva com menos pontos, mantendo as características essenciais da forma dentro de uma tolerância especificada.

O processo começa conectando os primeiros e últimos pontos da curva com uma linha, identificando o ponto mais distante desta linha e mantendo-o se sua distância exceder a tolerância. Este ponto divide a curva em dois segmentos, e o algoritmo simplifica recursivamente cada segmento. Esta abordagem iterativa reduz significativamente a complexidade dos dados em aplicações de SIG, melhorando a eficiência de armazenamento, processamento e exibição sem comprometer a integridade visual das características geográficas.

O algoritmo Ramer-Douglas-Peucker exemplifica como a redução de dados pode ser alcançada sem perder detalhes geométricos significativos, tornando-se inestimável em mapeamento digital e várias aplicações que requerem a renderização eficiente de formas complexas.
[![Watch the video](https://img.youtube.com/vi/u1ZMzY5kwiA/0.jpg)](https://youtu.be/u1ZMzY5kwiA)

---

## Instalação

Para instalar as dependências necessárias, você precisa ter o Python instalado no seu sistema. Se você ainda não tem o Python, você pode baixá-lo [aqui](https://www.python.org/downloads/). Após a instalação do Python, siga os passos abaixo:

1. **Clone o Repositório**

   Primeiro, clone o repositório do Easy Curve Fit para a sua máquina local.


2. **Instale as Dependências**

   Dentro do diretório do projeto, existe um arquivo chamado `requirements.txt` que contém todas as bibliotecas necessárias. Para instalá-las, execute o seguinte comando: **pip install -r requirements.txt**
   

3. Isso vai instalar todas as dependências necessárias para rodar o Easy Curve Fit.

## Execução

Para executar a aplicação, siga estas etapas:

* Navegue até o diretório do projeto onde o `main.py` está localizado.

* Execute o arquivo `main.py` usando Python: **python main.py**

* Após executar o comando, o Dash vai iniciar o servidor local e você poderá acessar a aplicação através do seu navegador. Normalmente, a URL será algo como `http://127.0.0.1:8050/`.

## Exemplos de Datasets

* No diretório [Datasets](https://github.com/Spogis/EasyCurveFit/tree/master/Datasets) você encontrará datasets de exemplos que podem lhe auxiliar.

## Suporte

Se você encontrar algum problema ou tiver alguma dúvida, não hesite em abrir uma issue no repositório do GitHub ou entrar em contato conosco diretamente.

##  Contato: https://linktr.ee/CascaGrossaSuprema

---

Esperamos que você aproveite o uso do Easy Curve Fit!
