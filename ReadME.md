# **Решение профильного задания на стажировку Data Scientist от vk**

В задании было предложено решить задачу геоаналитики и предсказать успешность (score) объекта ритейла. 

*Целевая метрика: MAE*

В дополнение к исходным признакам, предложенным организаторами, мной было решено добавить информацию о регионе, в котором расположен объект: через API dadata для каждой координаты была получена информация о регионе, затем, с помощью открытого датасета минздрава о количестве населения в регионах россии, для каждого региона было определено количество населения в нем. 

Объединение данных из features и train/test проблем не вызвало, так как после несложного анализа данных стало ясно, что каждой точке из train/test соответствует единственная ближайшая к ней точка из features. Таким образом, проблему с несовпадением координат удалось решить поиском пары наиболее близких друг к другу точек.

Также был проведен отбор признаков, по итогам которого количество признаков для обучения было сокращено с 375 до 59
В качестве итоговой модели для предсказания была выбрана линейная модель HuberRegressor(alpha=0.0003, epsilon=1.2, max_iter=500, tol=5e-05) из библиотеки sklearn, оптимальное полученное значение метрики на тренировочном датасете - 0.0558.

**Описание файлов:**  
```test_task.zip``` - архив со всеми необходимыми для запуска файлами  
```solution.py``` - скрипт, генерирующий файл submission.csv  
```submission.csv``` - файл с предсказаниями на тестовой выборке test.csv  
```region_info.csv``` - файл с информацией о регионах, в которых расположены точки  
```feature_generator.py``` - скрипт, генерирующий файл region_info.csv с новыми признаками  
```coordinates_to_regions.csv``` - файл, в котором установлены соответствия между координатами и регионами  
```population.csv``` - файл с информацией о количестве населения в 2019 году в различных регионах  
```model.pkl``` - предобученнная модель
