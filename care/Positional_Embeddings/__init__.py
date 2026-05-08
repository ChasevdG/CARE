import os
import importlib
import pkgutil

# Get current module name, e.g., "models"
package_name = __name__
package_path = __path__[0]  # path to models/

for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
    if not is_pkg and not module_name.startswith("_"):
        importlib.import_module(f"{package_name}.{module_name}")