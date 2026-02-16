# Contributing to Mithridatium

Thank you for checking out **Mithridatium**! We are excited to have you here. This project is focused on building a framework to **verify the integrity of pretrained AI models** by detecting backdoors and data poisoning. If you are interested in contributing, below are some guidelines to help us make the most out of your contribution.

## Contribution Scope

⚠️ **Note:**

- Issues labeled **`internal team`** are reserved for the project’s assigned developers and will not be accepted from outside contributors.
- **`good first issue`** and **`help wanted`** indicate tasks that are open to the community.

We encourage you to watch this repository if you’d like to stay updated!

## Issue Labels

We use labels to organize contributions:

- **`internal team`** – restricted to the internal project developers.
- **`bug`**, **`enhancement`**, **`documentation`** – for categorizing tasks.
- In the future: **`good first issue`** and **`help wanted`** will indicate tasks that are open to the community.

## Getting Started

### How to Contribute

### 1. **Fork the Repository**

- Navigate to the Mithridatium repository on GitHub
- Click the "Fork" button on the top right of the repository page to create your own copy.

### 2. **Clone Your Fork**

- Clone the repository to your local machine:
  ```bash
  git clone https://github.com/<your-username>/mithridatium
  cd mithridatium
  ```

### 3. **Create a New Branch**

- Create a new branch for your contribution:
  ```bash
  git checkout -b feature-branch
  ```

---

## Contribution Workflow

### 1.Choose an Issue

- Choose an open issue from the GitHub project board and carefully read the details and acceptance criteria before starting to work on it.

### 2. Make Your Changes

- Work on your branch locally. Implement your changes and test them thoroughly to ensure they work correctly.
- For any CLI or defense module changes, please include examples of expected input/output in the pull request.

### 3. Commit Your Changes

- Commit messages should be clear and concise. Follow the format:
  ```
  git commit -m "Add feature X to improve performance"
  ```
- Make sure your commit is properly documented and explains the **why** and **what** of the changes.

### 4. Push Your Changes

- Push your branch to GitHub:
  ```bash
  git push origin feature-branch
  ```

### 5. Submit a Pull Request (PR)

- Navigate to your fork on GitHub and click the **Pull Request** button.
- Ensure your PR:
  - References the related issue number (e.g., `Fixes #123`).
  - Provides a clear description of what was changed and why.
  - Includes relevant tests or screenshots where applicable.
  - follow the Pull Request template.

---

## Code Guidelines

- Keep your code readable, maintainable, and well-documented.

### 1. Documentation

- Update documentation as necessary. If your change impacts functionality, be sure to update the corresponding documentation in the **Help** or **README** files.

### 1. Structure

- New defenses should be added inside the defenses/ folder with their own module.
- Tests for new features should be placed in the tests/ folder.

---

## Issue Reporting

### 1. Suggesting Enhancements

- If you have ideas for new features, improvements, or optimizations, submit them through a GitHub issue and tag it with **enhancement**.

### 2. Reporting Bugs

- IFile a GitHub issue labeled bug and describe how to reproduce the problem, expected vs. actual behavior, and environment details.

---

## Community

- If you have any questions or need guidance, feel free to reach out through GitHub issues or via email pelumi.oluwategbe@slu.edu

Thank you for your contributions!
