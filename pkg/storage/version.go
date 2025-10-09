package storage

import (
	"fmt"
)

// Version represents the serialization format version
type Version struct {
	Major int
	Minor int
	Patch int
}

// Current version of the serialization format
var CurrentVersion = Version{Major: 1, Minor: 0, Patch: 0}

// String returns the version as a string
func (v Version) String() string {
	return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
}

// IsCompatible checks if a version is compatible with the current version
// Backward compatibility rules:
// - Major version must match
// - Minor version can be lower or equal
// - Patch version doesn't affect compatibility
func (v Version) IsCompatible(other Version) bool {
	// Major version must match
	if v.Major != other.Major {
		return false
	}
	// Current version must be >= other version (can read older formats)
	if v.Minor < other.Minor {
		return false
	}
	return true
}

// Compare returns -1 if v < other, 0 if v == other, 1 if v > other
func (v Version) Compare(other Version) int {
	if v.Major != other.Major {
		if v.Major < other.Major {
			return -1
		}
		return 1
	}
	if v.Minor != other.Minor {
		if v.Minor < other.Minor {
			return -1
		}
		return 1
	}
	if v.Patch != other.Patch {
		if v.Patch < other.Patch {
			return -1
		}
		return 1
	}
	return 0
}

// VersionedHeader wraps data with version information
type VersionedHeader struct {
	Version Version
	Format  Format
}

// WriteVersionedHeader writes a versioned header
func WriteVersionedHeader(w Writer, format Format) error {
	header := VersionedHeader{
		Version: CurrentVersion,
		Format:  format,
	}
	return w.Encode(header)
}

// ReadVersionedHeader reads and validates a versioned header
func ReadVersionedHeader(r Reader) (*VersionedHeader, error) {
	var header VersionedHeader
	if err := r.Decode(&header); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	if !CurrentVersion.IsCompatible(header.Version) {
		return nil, fmt.Errorf("incompatible version: file=%s, current=%s",
			header.Version.String(), CurrentVersion.String())
	}

	return &header, nil
}

// MigrationFunc is a function that migrates data from one version to another
type MigrationFunc func(r Reader, w Writer) error

// VersionMigrator handles version migrations
type VersionMigrator struct {
	migrations map[string]MigrationFunc // key: "fromVersion->toVersion"
}

// NewVersionMigrator creates a new version migrator
func NewVersionMigrator() *VersionMigrator {
	return &VersionMigrator{
		migrations: make(map[string]MigrationFunc),
	}
}

// RegisterMigration registers a migration function
func (vm *VersionMigrator) RegisterMigration(from, to Version, fn MigrationFunc) {
	key := fmt.Sprintf("%s->%s", from.String(), to.String())
	vm.migrations[key] = fn
}

// Migrate performs migration from one version to another
func (vm *VersionMigrator) Migrate(from, to Version, r Reader, w Writer) error {
	if from.Compare(to) == 0 {
		return nil // No migration needed
	}

	key := fmt.Sprintf("%s->%s", from.String(), to.String())
	migrateFn, exists := vm.migrations[key]
	if !exists {
		return fmt.Errorf("no migration path from %s to %s", from.String(), to.String())
	}

	return migrateFn(r, w)
}
