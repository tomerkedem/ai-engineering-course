from llm_factory import build_llm


def validate_ingredients(ingredients: str) -> str:
    ingredients = ingredients.strip()

    if not ingredients:
        raise ValueError("Please provide at least one ingredient.")

    return ingredients


def generate_recipe_name(llm, ingredients: str) -> str:
    prompt = f"""
You are a creative recipe naming assistant.

Create one short recipe name based on these ingredients:
{ingredients}

Rules:
- Return only the recipe name.
- Do not include explanations.
- Do not include cooking instructions.
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def write_cooking_instructions(
    llm,
    ingredients: str,
    recipe_name: str,
) -> str:
    prompt = f"""
You are a practical cooking assistant.

Recipe name:
{recipe_name}

Ingredients:
{ingredients}

Write simple cooking instructions for this recipe.

Rules:
- Keep the instructions beginner-friendly.
- Use numbered steps.
- Do not add ingredients that were not provided unless absolutely necessary.
- Keep the answer concise.
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def run_recipe_workflow(llm, ingredients: str) -> str:
    ingredients = validate_ingredients(ingredients)

    recipe_name = generate_recipe_name(llm, ingredients)

    instructions = write_cooking_instructions(
        llm=llm,
        ingredients=ingredients,
        recipe_name=recipe_name,
    )

    return (
        f"Recipe name: {recipe_name}\n\n"
        f"Cooking instructions:\n{instructions}"
    )


def main():
    llm = build_llm()

    print("Recipe Workflow is ready.")
    print("Enter ingredients, or type 'quit' / 'exit' to stop.")

    while True:
        ingredients = input("\nIngredients: ").strip()

        if ingredients.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        try:
            result = run_recipe_workflow(llm, ingredients)
            print(f"\n{result}")

        except ValueError as error:
            print(f"\nError: {error}")

        except Exception:
            print("\nError: Something went wrong while running the workflow.")


if __name__ == "__main__":
    main()