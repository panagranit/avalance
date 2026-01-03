# MyProject

Quick workspace created for VS Code.

## Getting Started

If you've been away from the terminal for a while and need to restart the environment:

1. Navigate to the project directory:
   ```powershell
   cd C:\Users\grant\Projects\MyProject
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   Or use the activation script:
   ```powershell
   .\activate_env.ps1
   ```
   
   *Note: If you get an execution policy error, run this first:*
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. Run the application:
   ```powershell
   python app.py
   ```



