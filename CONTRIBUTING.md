# Contributing to software-design-project

## Code Style Guide

To maintain a consistent and readable codebase, use the following style rules.

### Capitalization (Naming Conventions)

* **`PascalCase`**: Use for `structs`, `enums`, and `classes`.
    * *Example:* `class UserSession;`, `struct UserData;`
* **`camelCase`**: Use for `variables` and `functions`.
    * *Example:* `int userAge;`, `void getUserName();`
* **`SCREAMING_SNAKE_CASE`**: Use for `constants` and `macros`.
    * *Example:* `const int MAX_USERS = 10;`

### Commenting

Use [DoxyGen](https://www.doxygen.nl/) for all in-code documentation. 

* **File Headers:** Every file must begin with a DoxyGen comment block describing its purpose, author, and date.
    ```cpp
    /**
     * @file
     * @brief A brief description of what this file contains.
     * @author [Your Name]
     * @date [Date]
     */
    ```
* **Code Elements:** All `classes`, `structs`, `enums`, `constructors`, `functions`, and public members must have a DoxyGen comment block explaining what they do, their parameters, and what they return.
    ```cpp
    /**
     * @brief Calculates the sum of two integers.
     * @param a The first integer.
     * @param b The second integer.
     * @return The sum of a and b.
     */
    int add(int a, int b);
    ```

## Development Workflow

Follow these steps to make changes to the project. **Never push directly to the `main` branch.**

### 1. Creating a New Branch

Before you start coding, create a new branch for your task. This keeps your work separate from the stable code.

1.  **Get the latest code:**
    ```bash
    git checkout main
    git pull
    ```
2.  **Create and switch to your new branch:**
    ```bash
    git checkout -b <branch-type>/<branch-name>
    ```
    * **Branch Types:** Use `feat` (feature), `fix` (bug fix), `docs`, or `test`.
    * **Example:** `git checkout -b feat/pawn-movement` or `git checkout -b fix/rendering-bug`

### 2. Making a Commit

Once you have saved your changes, follow these steps to commit them to your branch.

1.  **Check what files you changed:**
    ```bash
    git status
    ```
2.  **Stage your changes:**
    * To stage *all* changes:
      ```bash
      git add .
      ```
    * To stage specific files:
      ```bash
      git add src/MyFile.cpp
      ```
3.  **Commit your changes:**
    ```bash
    git commit -m "type(scope): description of change"
    ```
    *(See the [Commit Message Guidelines](#commit-message-guidelines) below for the format)*

4.  **Push your branch to GitHub:**
    ```bash
    git push -u origin <your-branch-name>
    ```

### 3. Merging to Main (Pull/Merge Requests)

Once your code is pushed to your branch, it needs to be reviewed by your teammate before it joins the `main` codebase.

1.  **Open a Merge Request:**
    * Go to the project page on GitLab/GitHub.
    * Click "Create Merge Request" for your recently pushed branch.
    * Ensure your branch is the *source* and `main` is the *target*.

2.  **Fill out the Description:**
    * Briefly explain *what* you changed so your teammate knows what to look for.
    * Link to any relevant issues (e.g., "Closes #12").

3.  **Request a Review:**
    * Since we are a small team, just let your teammate know the MR is ready.
    * **Rule of Thumb:** Try not to merge your own code without at least a "thumbs up" or a quick look from the other person. 

4.  **Merge:**
    * Once given the go-ahead, click the "Merge" button.
    * This will automatically merge your branch into `main`.

## Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard. 

Your commit message header should follow this format:
`<type>(<scope>): <subject>`

* **Type:** `feat` (new feature), `fix` (bug fix), `docs` (documentation), `style` (formatting), `refactor`, `test`, `chore` (build/tool changes).
* **Scope (optional):** The module or part of the code affected (e.g., `auth`, `parser`).
* **Subject:** A short, imperative-tense description (e.g., "add login button" not "added login button").

**Good Examples:**
* `feat(auth): add password reset endpoint`
* `fix(parser): correct handling of empty strings`
* `docs(contributing): add DoxyGen examples`

**Bad Examples:**
* `fixed a bug`
* `updated code`
* `stuff`
