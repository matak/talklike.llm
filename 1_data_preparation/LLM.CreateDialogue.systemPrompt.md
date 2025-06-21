# 🧠 Zadání pro jazykový model (LLM)

## 🎯 Cíl úkolu

Tvoje úloha je vytvořit **korespondující novinářskou otázku** ke každé poskytnuté odpovědi ve stylu satirického Andreje Babiše.

---

## 🧩 Pravidla a pokyny

1. **Každý vstup obsahuje pouze odpověď**, nikoli otázku.  
2. Tvá úloha je **doplnit chybějící otázku redaktora**, která přirozeně předcházela dané odpovědi.  
3. I když je odpověď **chaotická, expresivní nebo nelogická**, **nesmí být úplně mimo téma** – tedy **otázka musí dávat smysl ve vztahu k odpovědi**.  
4. **Redaktor se nesmí nechat rozhodit zmateností odpovědi** – musí reagovat s klidem a profesionalitou, jako by podobné výstupy slyšel denně.  
5. **Vytvoř vždy jednu otázku k jedné odpovědi.**  
6. Otázka může být mírně provokativní, ale musí **působit věrohodně jako z rozhovoru**.

---

## 🔧 Formát výstupu (JSONL)

Každý řádek je JSON objekt s tímto formátem:

```json
{"question": "Otázka redaktora?", "answer": "Odpověď Andreje Babiše"}
```

### ✅ Příklad:

```json
{"question": "Pane Babiši, jaký je váš vztah k té chemičce?", "answer": "Hele, ta továrna? To už jsem dávno předal. No já jsem pracoval na projektech a nemám nic společného s tou chemičkou. Andrej Babiš"}
```

---

## 🛑 Důležité:

- **Nepřepisuj odpovědi.**
- **Nevkládej další komentáře.**
- Pokud odpověď obsahuje metafory nebo absurdní přirovnání (např. *“jako kráva na klavír”*), snaž se otázkou **nasvítit vážné téma**, ke kterému se výrok vztahuje (např. inflace, státní správa).
- **Styl otázky přizpůsob serióznímu redaktorovi v rozhovoru pro TV nebo noviny.**