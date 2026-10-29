package hack.sber.repository;

import hack.sber.model.DatasetRecord;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;
import java.util.UUID;

public interface DatasetRepository extends JpaRepository<DatasetRecord, UUID> {

    Optional<DatasetRecord> findTopByOrderByCreatedAtDesc();
}