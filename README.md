# Statistics
* 7.csv файл с входными данными   
* график регрессии строится только для величин чья корреляция больше 0,7 или равна ей по модулю     
***функции внутри программы:***    
* read_update - чтение файла
* day_night - делит данные на день и ночь
* df_mounth_day_night
* wind - добавляет столбец, где описание направление ветра конвертировано в градусы
* cloud - добавляет столбец, где описание облачности конвертировано в проценты
* stat - считает статистические данные    
* korrel - считает корреляцию и строит график корреляции двух величин
* confidence_interval - считает доверительный интервал
* print_stat - выводит некоторые статистические данные
* gistogramms - строит гистограммы по данным за месяц для величины за день и ночь
* correl_matrix - строит корреляционные матрицы
* regres - строит график линейной регрессии и подсчитывает ее уравнение
