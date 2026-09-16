# Preparación visual del catálogo (Fase 6a)

Cómo se decidieron los `visual_group`, qué se fusionó y de dónde salen las fotos de
`assets/`. Es el paso previo a la demo de la Fase 6b, se ejecuta **una sola vez** y es la
única parte del proyecto que necesita internet.

```bash
python -m src.catalog.build_assets              # descarga solo lo que falte
python -m src.catalog.build_assets --force      # vuelve a descargar todo
python -m src.catalog.build_assets --offline    # regenera los CSV sin tocar la red
```

Se verifica con `pytest tests/test_catalog.py` (30 tests, ninguno llama a Pexels).

---

## Rehecha en la Fase 7e, sin llamar a Pexels

La Fase 7a redujo el catálogo de 1.500 a 496 productos (8 referencias por categoría) y
renumeró los `product_id`. Lo que eso cambió aquí, y lo que no:

- **Los 60 `visual_group` y sus términos de búsqueda son los mismos.** El mapa va por
  `category`, y las 62 categorías no cambiaron: `visual_groups.csv` sale byte-idéntico.
  Las 60 fotos y `image_credits.csv` se reutilizan tal cual; **0 llamadas a la API**.
- **`product_catalog.csv` sí había que regenerarlo**, y es el único fichero que cambia
  (1.500 → 496 filas). Se hizo con `python -m src.catalog.build_assets --offline`.
- **El fallo no se veía.** Los `product_id` son correlativos, así que el CSV viejo
  (`P00001`–`P01500`) seguía cruzando con los 496 ids nuevos, pero solo el 1,6 % caía en su
  grupo correcto: la demo pintaba nombres y fotos de otra categoría sin ningún error.
  Ahora lo impiden dos cosas: `check_catalog_matches` en `src/demo/catalog.py`, que hace
  fallar la demo con el comando que lo arregla, y
  `test_el_csv_final_corresponde_al_catalogo_actual`.
- **El umbral de fusión ya no discrimina.** Con 8 referencias por categoría, 58 de los 60
  grupos quedan por debajo de `MIN_GROUP_SIZE = 15`. Las dos fusiones de abajo se
  mantienen porque se sostienen en las condiciones que mandan (mismo objeto físico, mismo
  departamento), no en el tamaño; los recuentos de sus tablas son los del catálogo de
  1.500 con el que se decidieron.

---

## Qué se genera

| Fichero | Contenido |
| --- | --- |
| `assets/<visual_group>.jpg` | Una foto por grupo, recortada a 800x800 y comprimida |
| `assets/visual_groups.csv` | `visual_group, search_term` — el término de búsqueda de cada grupo |
| `assets/product_catalog.csv` | `product_id, product_name, visual_group, image_path` — el mapeo final, una fila por producto |
| `assets/image_credits.csv` | Autoría de cada foto: fotógrafo, URL en Pexels, texto alternativo, consulta usada y puntuación |

Todo esto se comitea. **No se regenera en cada ejecución del pipeline**: los resultados de
búsqueda de Pexels no son reproducibles por semilla (el catálogo de fotos cambia con el
tiempo), así que `assets/` se trata como un fixture cacheado, igual que dice `CLAUDE.md`
en la excepción de reproducibilidad. Relanzar el script sin `--force` no vuelve a pedir a
Pexels ninguna foto que ya esté descargada.

---

## Por qué `category` y no otra columna

`products` solo tiene tres columnas descriptivas, y ninguna otra sirve:

| Columna | Valores | Por qué no |
| --- | ---: | --- |
| `department` | 8 | Demasiado amplio: una sola foto para "Frescos" tendría que representar a la vez leche, pan, fruta, carne y pescado. |
| `brand` | 133 | Son razones sociales generadas por Faker ("Familia Mir S.A."), sin aspecto propio: dos marcas de leche no se ven distintas. |
| `product_id` | 496 | El SKU: 496 fotos distintas para un dataset sintético, y ninguna diría nada que no diga ya la categoría. |
| **`category`** | **62** | El nivel que pide el reto: "Leche", "Yogur", "Pescado blanco" son justo el tipo de grupo reutilizable del ejemplo (`leche_entera`, `yogur_griego`, `salmón`). |

**`visual_group` no parte la categoría en trozos más finos**, y esto es una limitación
consciente del dataset, no un descuido: no hay ninguna columna de variedad, formato o
sabor con la que separar "leche entera" de "leche desnatada". Inventarla sería fabricar un
dato que el generador nunca escribió. Si en algún momento se quiere ese grano, el cambio
empieza en `DATA_SPEC.md` y en el generador, no aquí.

Aun así `visual_group` **es una columna propia y no un alias** de `category`:

1. Es un slug ASCII en `snake_case`, usable como nombre de fichero (`assets/leche.jpg`) y
   estable frente a la grafía sucia de la categoría (`category_raw` trae `LECHE`, `leche`,
   `Leche `...).
2. Dos categorías se fusionan con una hermana, así que el mapa es **62 → 60**, no 1:1.

---

## Fusiones

### El criterio

Se fusiona una categoría solo si cumple las tres condiciones:

1. tiene menos de **15 productos** (`MIN_GROUP_SIZE`),
2. existe una hermana cuya foto describe con honestidad *todos* los productos de la
   pequeña — el mismo objeto físico, no solo "temas parecidos",
3. las dos están en el mismo departamento.

La condición (2) es la que manda. El objetivo de fusionar no es ahorrar descargas (60
llamadas a una API gratuita no son un problema), sino **evitar dos grupos que acabarían
con la misma foto**: en la demo, dos tarjetas distintas con la misma imagen se leen como
un error.

### Lo que se fusionó

| Categoría absorbida | Productos | Grupo destino | Por qué |
| --- | ---: | --- | --- |
| `Bacalao` | 10 | `pescado_blanco` | El bacalao **es** un pescado blanco: la misma foto de lomo crudo describe con honestidad las dos categorías. Mismo departamento (Frescos) que `Pescado blanco` (22). La estacionalidad de Cuaresma no se pierde: vive en `category`, no en la imagen. |
| `Limpiacristales` | 15 | `limpiadores_hogar` | Limpiacristales y `Lejía y limpiadores` (21) son el mismo objeto en la estantería: botella o spray de limpiador doméstico. Por separado acababan con dos fotos prácticamente iguales. |

### Lo que **no** se fusionó, siendo igual de pequeño

Cinco grupos se quedan por debajo del umbral y aun así siguen solos, porque ninguna
hermana los representa sin mentir. Es la parte del criterio que conviene tener presente al
leer el recuento:

| Grupo | Productos | Por qué sigue solo |
| --- | ---: | --- |
| `turron` | 7 | El turrón es una barra de nougat; la hermana natural por tamaño sería `chocolate`, y una foto de chocolate no es turrón. Además es el producto que carga toda la estacionalidad de diciembre (multiplicador 8,0x): fundirlo con chocolate borra el pico de Navidad de la demo. |
| `torrijas` | 8 | Bollería dulce de Cuaresma. La única hermana en Frescos es `pan`, y una hogaza no describe una torrija. |
| `cava` | 10 | La hermana sería `vino`, pero la botella de espumoso tiene forma, morro y tapón propios. Es también el producto de diciembre junto al turrón. |
| `protector_solar` | 10 | Está en Droguería, rodeado de productos de limpieza: no tiene ninguna hermana visual. Es el pico de verano (6,0x). |
| `marisco` | 13 | Está en Congelados, entre pizzas, helados y verduras. `pescado_blanco` sería lo más cercano, pero está en otro departamento y un lomo de pescado no es una gamba. |

Los cinco son, además, los productos que sostienen las historias de estacionalidad que el
generador inyecta a propósito (`DATA_SPEC.md`, "Estacionalidad"). Fusionarlos habría sido
ahorrar cinco fotos a cambio de aplanar justo lo que la demo debería poder enseñar.

---

## Cómo se eligen las fotos

Pexels no permite pedir "producto sobre fondo blanco, sin personas", así que la selección
se hace en tres pasos (`src/catalog/pexels.py`):

1. **Consulta en inglés.** El término de búsqueda va en inglés aunque el resto del
   proyecto esté en español: es donde Pexels tiene cobertura. Se lanzan tres variantes
   —la consulta completa con `color=white`, la misma sin filtro de color, y una versión
   corta— y todas las candidatas caen en un mismo pozo del que se elige la mejor. Usar
   `color=white` en cascada, como primera versión del script, sacaba sobre todo envases
   en blanco de mockup: limpísimos y sin ningún producto reconocible.
2. **Puntuación sobre el texto alternativo y el color medio.** Descarta las fotos con
   personas (`woman`, `crop`, `anonymous`, `lying`, partes del cuerpo...), suma por
   luminosidad del color medio y por palabras de estudio (`isolated`, `white background`,
   `packaging`), y resta por escenas (`restaurant`, `kitchen`, `street`), bodegones de
   varios objetos (`with`, `stacked`, `arrangement`) y envases en blanco (`mockup`,
   `blank`).
3. **Huella perceptual antes de guardar** (`src/catalog/image_hash.py`). El `id` de Pexels
   no basta: hay fotos distintas de la misma sesión que llegan con id distinto y aspecto
   idéntico. Se calcula un dHash de cada candidata descargada y se rechaza si se parece a
   la de otro grupo, bajando por la lista de candidatas hasta encontrar una que no se
   repita.

El punto 3 apareció resolviendo un caso concreto: `detergente` y `suavizante` —que el
dataset relaciona con un lift de 3,5x, así que la demo los va a enseñar juntos como
co-compra— se estaban llevando dos fotos del mismo bote rosa.

### La heurística es eso, una heurística

El filtro por texto alternativo acierta a menudo pero no siempre, y **la revisión final fue
visual**: se montó un contact sheet con las 60 fotos y se relanzaron a mano los grupos que
habían salido mal. Hicieron falta tres tandas. Ejemplos de lo que hubo que corregir:

- `panales` cayó dos veces en fotos de bebés (*"baby's feet"*, *"baby lying on back"*) — de
  ahí que la lista de descarte incluya partes del cuerpo y posturas, y no la palabra
  `baby`, que es la palabra clave legítima de cuatro grupos.
- `arena_gato` y `bolsas_basura` se llevaron la misma foto de papel arrugado.
- `yogur` salió con un vaso de leche, `potitos` con un tarro de curry, `sopas` con dos
  latas en blanco.

Como las fotos quedan comiteadas y a la vista, cualquiera puede repetir esa revisión.
Si un grupo se ve mal, se borra su `.jpg`, se ajusta su término en
`CATEGORY_TO_GROUP` y se relanza el script: solo vuelve a pedir lo que falta.

### Grupos que se aceptaron sabiendo que no son perfectos

| Grupo | Qué tiene | Por qué se acepta |
| --- | --- | --- |
| `leche_infantil` | Una lata blanca sin etiqueta | Es un bote de formato correcto; Pexels no tiene stock limpio de leche infantil que no salga con un bebé. |
| `harina` | Textura de harina, no un paquete | Reconocible como harina, aunque no como producto empaquetado. |
| `verdura` | Solo cebollas | Es una verdura sobre fondo limpio; las fotos de "verdura variada" venían en escenas de mercado. |
| `suavizante` | Bote rosa parecido al de `detergente` | Son fotos distintas (distancia dHash 19, muy lejos del umbral de 8), pero de la misma familia visual. Es el par de co-compra del dataset, así que conviene saberlo. |

---

## Nombre de producto

El dataset **no tiene un nombre de producto propio** (`DATA_SPEC.md`), así que
`product_name` se compone en esta fase, sin tocar el esquema ni el generador:

```
{category} {brand sin forma jurídica} [- Pack {pack_size} si > 1]
```

- `Familia Mir S.A.` → `Familia Mir`, `Comercial Bru y asociados S.L.L.` → `Comercial Bru`.
- `Leche Familia Mir`, `Leche Comercial Bru - Pack 6`.
- Categoría, marca y formato no bastan para 496 SKU distintos (10 productos con
  nombre repetido; eran 177 sobre los 1.500 del catálogo anterior a la Fase 7a), así que
  los nombres repetidos se numeran: `Leche Familia Mir (1)`, `Leche Familia Mir (2)`. En un
  lineal real serían variedades; aquí basta con que la demo no muestre dos tarjetas con el
  mismo título.

---

## El recuento completo

60 grupos, 496 productos: 8 por grupo, salvo los dos fusionados, que suman 16. Hasta la
Fase 7a eran 1.500 productos y 25 por grupo de media (mínimo 7, máximo 41); los grupos y
los términos de búsqueda no han cambiado.

| `visual_group` | Departamento | Categorías que agrupa | Productos | Término de búsqueda |
| --- | --- | --- | ---: | --- |
| `leche_infantil` | Bebe | Leche infantil | 8 | baby formula powder tin white background |
| `panales` | Bebe | Panales | 8 | disposable diapers stack folded |
| `potitos` | Bebe | Potitos | 8 | jars of baby food puree row |
| `toallitas` | Bebe | Toallitas humedas | 8 | wet wipes tissue pack |
| `agua` | Bebidas | Agua | 8 | bottled water plastic bottle white background |
| `cava` | Bebidas | Cava y espumosos | 8 | sparkling wine bottle and glass |
| `cerveza` | Bebidas | Cerveza | 8 | beer bottles isolated white background |
| `refrescos` | Bebidas | Refrescos | 8 | aluminium soft drink cans stack |
| `vino` | Bebidas | Vino | 8 | red wine bottle isolated white background |
| `zumos` | Bebidas | Zumos | 8 | orange juice bottle white background |
| `helados` | Congelados | Helados | 8 | ice cream tub isolated white background |
| `marisco` | Congelados | Marisco | 8 | raw prawns shrimp white background |
| `pizza` | Congelados | Pizza congelada | 8 | frozen pizza isolated white background |
| `precocinados` | Congelados | Precocinados congelados | 8 | frozen ready meal package white background |
| `verduras_congeladas` | Congelados | Verduras congeladas | 8 | frozen peas vegetables white background |
| `aceite_oliva` | Despensa | Aceite de oliva | 8 | olive oil bottle white background |
| `arroz` | Despensa | Arroz | 8 | uncooked white rice grains in bowl |
| `azucar` | Despensa | Azucar y edulcorante | 8 | white sugar cubes in bowl |
| `cafe` | Despensa | Cafe | 8 | coffee beans package white background |
| `cereales` | Despensa | Cereales | 8 | breakfast cereal box white background |
| `chocolate` | Despensa | Chocolate y huevos de Pascua | 8 | dark chocolate bar squares broken |
| `conservas_pescado` | Despensa | Conservas de pescado | 8 | canned tuna tin white background |
| `especias` | Despensa | Sal y especias | 8 | spice jars isolated white background |
| `galletas` | Despensa | Galletas | 8 | biscuits cookies isolated white background |
| `harina` | Despensa | Harina | 8 | flour bag baking white background |
| `legumbres` | Despensa | Legumbres | 8 | dried beans lentils white background |
| `palomitas` | Despensa | Palomitas de microondas | 8 | popcorn bowl isolated white background |
| `pasta` | Despensa | Pasta | 8 | dry spaghetti pasta white background |
| `salsa_tomate` | Despensa | Salsa de tomate | 8 | tomato sauce jar white background |
| `snacks` | Despensa | Snacks y aperitivos | 8 | potato chips snack bag white background |
| `sopas` | Despensa | Sopas y caldos | 8 | vegetable soup can tin label |
| `turron` | Despensa | Turron y mazapan | 8 | nougat almond bar sweet white background |
| `bolsas_basura` | Drogueria | Bolsas de basura | 8 | roll of garbage bags plastic isolated |
| `detergente` | Drogueria | Detergente | 8 | laundry detergent bottle white background |
| `lavavajillas` | Drogueria | Lavavajillas | 8 | dishwasher detergent tablets white background |
| **`limpiadores_hogar`** | Drogueria | **Lejia y limpiadores + Limpiacristales** | 16 | cleaning spray bottles white background |
| `protector_solar` | Drogueria | Protector solar | 8 | sunscreen bottle isolated white background |
| `suavizante` | Drogueria | Suavizante | 8 | fabric softener blue bottle laundry care |
| `carne_pollo` | Frescos | Carne de pollo | 8 | raw chicken breast white background |
| `carne_ternera` | Frescos | Carne de ternera | 8 | raw beef steak white background |
| `embutido` | Frescos | Embutido y fiambre | 8 | sliced cured ham charcuterie white background |
| `fruta` | Frescos | Fruta | 8 | fresh fruit assortment white background |
| `huevos` | Frescos | Huevos | 8 | eggs carton isolated white background |
| `leche` | Frescos | Leche | 8 | milk bottle isolated white background |
| `pan` | Frescos | Pan | 8 | loaf of bread isolated white background |
| **`pescado_blanco`** | Frescos | **Bacalao + Pescado blanco** | 16 | raw white fish fillet white background |
| `queso` | Frescos | Queso | 8 | cheese wedge isolated white background |
| `torrijas` | Frescos | Torrijas y bolleria de Cuaresma | 8 | sweet pastry bun sugar white background |
| `verdura` | Frescos | Verdura | 8 | fresh vegetables isolated white background |
| `yogur` | Frescos | Yogur | 8 | greek yogurt bowl with spoon |
| `acondicionador` | Higiene | Acondicionador | 8 | hair conditioner bottle white background |
| `champu` | Higiene | Champu | 8 | shampoo bottle isolated white background |
| `desodorante` | Higiene | Desodorante | 8 | deodorant spray can cosmetic product |
| `gel_ducha` | Higiene | Gel de ducha | 8 | body wash shower gel bathroom bottle |
| `higiene_femenina` | Higiene | Higiene femenina | 8 | menstrual pads tampons box product |
| `papel_higienico` | Higiene | Papel higienico | 8 | toilet paper rolls white background |
| `pasta_dientes` | Higiene | Pasta de dientes | 8 | toothpaste tube white background |
| `arena_gato` | Mascotas | Arena para gato | 8 | cat litter granules pellets |
| `comida_gato` | Mascotas | Comida para gato | 8 | cat food kibble bowl white background |
| `comida_perro` | Mascotas | Comida para perro | 8 | dry dog food kibble pile |

---

## Licencia de las fotos

Todas vienen de [Pexels](https://www.pexels.com) bajo su
[licencia gratuita](https://www.pexels.com/license/), que permite uso comercial y no exige
atribución. Aun así, `assets/image_credits.csv` guarda fotógrafo y URL de cada una, que es
lo que Pexels recomienda y lo mínimo razonable en un proyecto que se enseña.

Ninguna imagen está generada por IA: son fotografías de stock, como pide `CHALLENGE.md`.

## La clave de la API

`PEXELS_API_KEY` se lee de `.env` (ignorado por git) con `python-dotenv`. `.env.example`
sí se comitea, sin valor real, como documentación de qué variable hace falta. La clave no
se imprime nunca: los mensajes de error hablan de la variable, no de su valor.
