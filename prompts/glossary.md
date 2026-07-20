# Cooking Glossary — EN → TR

Terminology lock for the translation stage. Every term below is drawn from a frequency scan of the 500 scraped answers, so this covers what actually appears in the corpus rather than generic cooking vocabulary.

Paste this file into the translation prompt (see `translate_tr.md`). Its job is consistency: without it, the same English term gets three different Turkish renderings across 500 examples, which is exactly the kind of noise that makes a fine-tuned model's vocabulary unstable.

> **Review this before using it.** These are proposed mappings, not authority. Adjust anything that reads wrong to a native ear — especially the ambiguous pairs at the bottom, where the right choice depends on context.

## Core

| English          | Turkish                                      |
| ---------------- | -------------------------------------------- |
| water            | su                                           |
| food             | yiyecek (gıda in safety/regulatory contexts) |
| cooking          | pişirme                                      |
| to cook          | pişirmek                                     |
| recipe           | tarif                                        |
| ingredient       | malzeme                                      |
| kitchen          | mutfak                                       |
| dish (the food)  | yemek                                        |
| taste (noun)     | tat                                          |
| flavor           | aroma                                        |
| texture          | doku                                         |
| temperature      | sıcaklık                                     |
| room temperature | oda sıcaklığı                                |
| heat (noun)      | ısı                                          |
| to heat          | ısıtmak                                      |
| to cool          | soğutmak                                     |
| time             | süre                                         |
| amount           | miktar                                       |

## Ingredients

| English       | Turkish       |
| ------------- | ------------- |
| salt          | tuz           |
| sugar         | şeker         |
| flour         | un            |
| egg           | yumurta       |
| milk          | süt           |
| butter        | tereyağı      |
| cream         | krema         |
| cheese        | peynir        |
| meat          | et            |
| chicken       | tavuk         |
| bread         | ekmek         |
| dough         | hamur         |
| pasta         | makarna       |
| garlic        | sarımsak      |
| onion         | soğan         |
| beans         | fasulye       |
| vegetables    | sebzeler      |
| chocolate     | çikolata      |
| coffee        | kahve         |
| tea           | çay           |
| yeast         | maya          |
| baking powder | kabartma tozu |
| baking soda   | karbonat      |
| stock / broth | et suyu       |
| sauce         | sos           |
| vinegar       | sirke         |

## Equipment

| English         | Turkish         |
| --------------- | --------------- |
| pan             | tava            |
| pot             | tencere         |
| oven            | fırın           |
| stove / hob     | ocak            |
| microwave       | mikrodalga      |
| fridge          | buzdolabı       |
| freezer         | dondurucu       |
| bowl            | kase            |
| knife           | bıçak           |
| lid             | kapak           |
| cast iron       | döküm demir     |
| non-stick       | yapışmaz        |
| pressure cooker | düdüklü tencere |
| thermometer     | termometre      |

## Techniques

| English               | Turkish                 |
| --------------------- | ----------------------- |
| to boil               | kaynatmak               |
| boiling               | kaynama                 |
| to simmer             | kısık ateşte pişirmek   |
| to bake               | fırında pişirmek        |
| to roast              | rosto yapmak / kavurmak |
| to fry                | kızartmak               |
| to deep-fry           | derin yağda kızartmak   |
| to sauté              | sotelemek               |
| to sear               | mühürlemek              |
| to grill              | ızgara yapmak           |
| to steam              | buharda pişirmek        |
| to poach              | poşe etmek              |
| to blanch             | haşlamak                |
| to knead              | yoğurmak                |
| to whisk / beat       | çırpmak                 |
| to marinate           | marine etmek            |
| to steep / brew       | demlemek                |
| to season             | tatlandırmak            |
| to rest (meat, dough) | dinlendirmek            |
| to proof (dough)      | mayalandırmak           |
| to thaw / defrost     | buzunu çözmek           |
| to store              | saklamak                |

## Food science

| English           | Turkish                   |
| ----------------- | ------------------------- |
| Maillard reaction | Maillard reaksiyonu       |
| caramelization    | karamelizasyon            |
| gelatinization    | jelatinizasyon            |
| emulsion          | emülsiyon                 |
| to emulsify       | emülsiyon hâline getirmek |
| to curdle         | kesilmek                  |
| fermentation      | fermantasyon              |
| oxidation         | oksidasyon                |
| starch            | nişasta                   |
| protein           | protein                   |
| gluten            | gluten                    |
| acid / acidity    | asit / asitlik            |
| pH                | pH                        |
| moisture          | nem                       |
| steam (noun)      | buhar                     |
| bacteria          | bakteri                   |
| food safety       | gıda güvenliği            |
| danger zone       | tehlike bölgesi           |
| shelf life        | raf ömrü                  |
| to spoil          | bozulmak                  |
| raw               | çiğ                       |
| cooked            | pişmiş                    |

## Measurements

Keep numerals and unit symbols as-is (`180 °C`, `350 °F`, `2 kg`). Translate
only spelled-out unit names.

| English              | Turkish                |
| -------------------- | ---------------------- |
| cup                  | su bardağı             |
| tablespoon           | yemek kaşığı           |
| teaspoon             | çay kaşığı             |
| pinch                | tutam                  |
| ounce                | ons                    |
| pound                | libre                  |
| Fahrenheit / Celsius | Fahrenheit / Santigrat |

## Ambiguous pairs — pick by context

These are the ones that will silently degrade the dataset if translated mechanically. Each English word maps to more than one Turkish word.

| English        | Turkish                    | Rule                                                                                                            |
| -------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------- |
| oil / fat      | yağ                        | Both are `yağ`. Use `sıvı yağ` for liquid oils, `katı yağ` for solid fats when the distinction carries meaning. |
| rice           | pirinç / pilav             | `pirinç` = the uncooked grain. `pilav` = the cooked dish. Never interchangeable.                                |
| dish           | yemek / tabak              | `yemek` = the prepared food. `tabak` = the physical plate.                                                      |
| to bake        | pişirmek / fırınlamak      | `fırında pişirmek` when the oven is the point; plain `pişirmek` otherwise.                                      |
| brown (verb)   | kızartmak / esmerleştirmek | `kızartmak` for frying-browning; `esmerleştirmek` for Maillard colour change.                                   |
| cover          | kapatmak / örtmek          | `kapağını kapatmak` for lidding a pot; `örtmek` for covering with foil/cloth.                                   |
| stock          | et suyu / stok             | `et suyu` in every culinary sense. `stok` only means inventory — never use it for broth.                        |
| dry (adj/verb) | kuru / kurutmak            | `kuru` describes state; `kurutmak` is the action.                                                               |
