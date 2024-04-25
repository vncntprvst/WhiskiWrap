#!/usr/bin/env python

import os
import stat
import distutils.sysconfig

def update_permissions(verbose=False):
    # Locate the 'bin' directory in the site-packages
    site_packages_dir = distutils.sysconfig.get_python_lib()
    bin_dir = os.path.join(site_packages_dir, 'whisk', 'bin')

    if not os.path.exists(bin_dir):
        print("Error: 'bin' directory not found. Please ensure the package is installed correctly.")
        return

    #  Test if trace can be executed
    trace_file = os.path.join(bin_dir, 'trace')
    if not os.access(trace_file, os.X_OK):
        print("Updating permissions of executable files in the 'whisk/bin' directory")
        for filename in os.listdir(bin_dir):
            file_path = os.path.join(bin_dir, filename)

            if os.path.isfile(file_path):
                # If on Windows skip files that are not executable
                if os.name == 'nt' and not filename.endswith('.exe'):
                    continue
                # If on Linux, skip files that have an extension
                elif os.name == 'posix' and filename.find('.') != -1:
                    continue

                # For current user: read, write, execute; for group and others: read only
                os.chmod(file_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
                # For all users: read, write, execute
                # os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                if verbose:
                    print(f"Updated permissions for {filename}")
        if verbose:
            print("Permissions update complete.")

if __name__ == "__main__":
    update_permissions()
