# Настройка окружения для курсов

Доступно два способа: использование образа преднастроенной виртуальной машины 
или установка пакетов на нативную систему с нуля.

## Виртуальная машина

1. Установите VirtualBox.
2. Получите OVA архив с преднастроенной системой (раздается на первом занятии).
3. Следуйте инструкция по [ссылке](http://askubuntu.com/questions/588426/how-to-export-and-import-virtualbox-vm-images) чтобы импортировать образ.

Вместо VirtualBox можно использовать любую систему виртуализации поддерживающую 
импорт из OVA, но проверялось все на VirtualBox.

Имя пользователя в системе: ds
Пароль: DScourses2017

## Ручная инсталляция на Linux (Ubuntu)

### Установка python

```bash
sudo apt-get install python-pip python-dev python-tk
```

### Обновление pip

```bash
pip install --upgrade pip
```

### Установка необходимых библиотек

Во всех нижеследующих пунктах если у вас есть настроенный virtualenv - можете 
создать новое виртуальное окружение и устанавливать в нём опуская ключ `--user`

Чтобы установить scipy с зависимостями следуйте инструкциям по [ссылке](https://www.scipy.org/install.html).

В простейшем случая вам понадобится лишь выполнить
```bash
pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

Установите необходимые библиотеки работы с данными и машинного обучения.
```bash
pip install --user statsmodel (опционально, отсутствие не критично)
pip install --user seaborn
pip install --user BeautifulSoup4
pip install --user scikit-learn 
pip install --user requests
```

Следующие библиотеки работы с нейронными сетями и их зависимости будут 
использоваться на последних уроках, если с ними возникнут проблемы - они 
решатся по ходу занятий.

```bash
pip install --user Theano
pip install --user tensorflow
pip install --user pillow
pip install --user h5py
pip install --user keras
```